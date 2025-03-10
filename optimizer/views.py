import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import re

from django.contrib.auth.models import User
from django.http import JsonResponse
from django.http import HttpResponse
from django.conf import settings
from django.contrib.auth import authenticate, login
from rest_framework.exceptions import ValidationError
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from .seializers import ImageUploadSerializer
import os
import cv2
import logging
logger = logging.getLogger(__name__)
import numpy as np
from .forms import SignUpForm, LoginForm
from django.shortcuts import render, redirect


def home_view(request):
    return render(request, 'home_optimize.html')

def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.refresh_from_db()
            user.profile.first_name = form.cleaned_data.get('first_name')
            user.profile.last_name = form.cleaned_data.get('last_name')
            user.profile.email = form.cleaned_data.get('email')

            return render(request, 'request_successful.html', {'user': user})
        else:
                return render(request, 'signup.html', {'form': form})
    else:
            form = SignUpForm()
    return render(request, 'signup.html', {'form': form})


def login_view(request):
     if request.method == 'POST':
        form = LoginForm(request=request, data= request.POST)
        if form.is_valid():
             username = form.cleaned_data.get('username')
             password = form.cleaned_data.get('password')
             user = authenticate(username=username, password=password)
             if user is not None:
                  login(request, user)
                  return redirect('schema-swagger-ui')
             else:
                form.add_error(None, "Invalid login credentials.")
        else:
            return render(request, 'login.html', {'form': form})

     else:
        form = LoginForm()

        return render(request, 'login.html', {'form': form})
     
class ObtainJWTTokenView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")

        user = authenticate(username=username, password=password)

        if user is not None:
            refresh = RefreshToken.for_user(user)
            return JsonResponse({
                'access_token': str(refresh.access_token),
                'refresh_token': str(refresh),
            }, status=status.HTTP_200_OK)
        
        return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)



class OptimizeImageView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = ([MultiPartParser])

    def validate_images(self, image, image2):
        if not image and not image2:
            raise ValidationError('At least one image must be provided')
        
        images = {}
        if image:
            images['img1'] = self.decode_single_image(image)
        if image2:
            images['img2'] = self.decode_single_image(image2)
        return images

    def decode_single_image(self, image):
        """Decode a single image file to OpenCV format"""
        file_data = image.read()
        nparr = np.frombuffer(file_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValidationError("Image could not be decoded")
        return img

    def process_image(self, img):
        """Process image with selected operations"""
        try:
            
            logger.info("Starting image processing")

            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if self.denoise:
                img = cv2.fastNlMeansDenoisingColored(img)

            if self.edge_detection:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(img_gray, 100, 200)
                img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            if self.cropping:
                x_start, y_start, x_end, y_end = map(int, self.cropping.split(','))
                img = img[y_start:y_end, x_start:x_end]

            if self.rotation_angle:
                rotation_angle = float(self.rotation_angle)
                rows, cols = img.shape[:2]
                center = (cols / 2, rows / 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                img = cv2.warpAffine(img, rotation_matrix, (cols, rows))

            if self.width and self.height:
                width = int(self.width)
                height = int(self.height)
                img = cv2.resize(img, (width, height))

            if self.gaussian_blur:
                kernel_size = int(self.gaussian_blur)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

            if self.contrast or self.brightness:
                contrast = float(self.contrast) if self.contrast else 1.0
                brightness = int(self.brightness) if self.brightness else 0
                img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

            if self.histogram_equalization:
                if len(img.shape) == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.equalizeHist(img_gray)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img = cv2.equalizeHist(img)

            if self.corner_detection:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = np.float32(gray)
                dst = cv2.cornerHarris(gray, 2, 3, 0.04)
                dst = cv2.dilate(dst, None)
                img[dst > 0.01 * dst.max()] = [0, 0, 255]

            if self.translate_x or self.translate_y:
                M = np.float32([[1, 0, self.translate_x], [0, 1, self.translate_y]])
                img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            if self.scale_x != 1.0 or self.scale_y != 1.0:
                img = cv2.resize(img, None, fx=self.scale_x, fy=self.scale_y, interpolation=cv2.INTER_LINEAR)

            if self.identify_features and self.img2 is not None:
                logger.info("شروع پردازش identify_features")
                img = self.identify_features_function(img, self.img2)


            if self.aligned_images and self.img2 is not None:
                img = self.aligned_images(img, self.img2)
            
            if self.combine_images and self.img2 is not None:
                img = self.combine_images(img, self.img2)

            if self.perspective_correction:
                img = self.perspective_correction(img)

            if self.kalman_line_detection:
                logger.info("Applying Kalman line detection")
                img = self.kalman_line_detection_function(img)  # فراخوانی تابع kalman_line_detection_function
                logger.info("Kalman line completed")

                
            if self.ransac_detection:
                logger.info("Applying ransac line detection")
                img = self.ransac_line_detection(img)
                logger.info("Kalman line detection completed")


            if self.curve_detection:
                logger.info("Applying curve line detection")
                img = self.curve_detection_function(img)
                logger.info("curve line detection detection")


            if self.optimize_parameters:
                logger.info("Applying optimize parameters detection")
                img = self.optimize_parameters_function(img)
                logger.info("optimize parameters detection detection")

            if self.enhance_image_quality:
                logger.info("Applying enhance image quality detection")
                img = self.enhance_image_quality_function(img)
                logger.info("enhance image quality detection detection")

            
            if self.detect_numbers:
                logger.info("Detecting numbers in image")
                img = self.detect_numbers_function(img)
                logger.info(f"Detected numbers: {self.detected_numbers}")


            return img

        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
        

    def detect_numbers_function(self, img):
        """Extract numbers from image using OCR"""
        result = img.copy()
        
        # تبدیل به grayscale
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # اعمال threshold برای جداسازی بهتر اعداد
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # استفاده از Tesseract برای تشخیص متن
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        
        # استخراج اعداد از متن
        numbers = re.findall(r'\d+', text)
        numbers = [int(num) for num in numbers]
        
        # رسم کادر دور اعداد در تصویر
        d = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
        n_boxes = len(d['level'])
        
        for i in range(n_boxes):
            if int(d['conf'][i]) > 60:  # اطمینان از صحت تشخیص
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        self.detected_numbers = numbers

        return result




    def enhance_image_quality_function(self, img):
        esult = img.copy()
    
        # 1. ابتدا نویزگیری هوشمند با حفظ لبه‌ها
        result = cv2.bilateralFilter(result, 9, 75, 75)
        
        # 2. تنظیم کنتراست و روشنایی با مقادیر دقیق
        result = cv2.convertScaleAbs(result, alpha=1.15, beta=0)
        
        # 3. بهبود جزئیات با Unsharp Masking
        gaussian = cv2.GaussianBlur(result, (0, 0), 2.0)
        result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)
        
        # 4. افزایش وضوح با حفظ کیفیت
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        result = cv2.filter2D(result, -1, kernel)
        
        # 5. افزایش رزولوشن با الگوریتم مناسب
        height, width = result.shape[:2]
        result = cv2.resize(result, (int(width*1.2), int(height*1.2)), 
                        interpolation=cv2.INTER_CUBIC)
        
        return result

        
    def kalman_line_detection_function(self, img):
        result = img.copy()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                                minLineLength=self.min_line_length, 
                                maxLineGap=self.max_line_gap)
        if lines is None:
            return result
        
        for line in lines:
                x1, y1, x2, y2 = line[0]
                prediction = kalman.predict()
                measurement = np.array([[x1], [y1]], np.float32)
                kalman.correct(measurement)
                
                pred_x = int(prediction[0])
                pred_y = int(prediction[1])
                cv2.line(result, (x1, y1), (pred_x, pred_y), (0, 255, 0), 2)
                cv2.line(result, (x2, y2), (pred_x, pred_y), (0, 0, 255), 2)
        
        return result


    def ransac_line_detection(self, img):
        # کپی تصویر
        result = img.copy()
        
        # تبدیل به grayscale و پیش‌پردازش
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # یافتن نقاط لبه
        points = np.column_stack(np.where(edges > 0))
        
        if len(points) < 2:
            return result
            
        best_line = None
        best_inliers = []
        
        for _ in range(self.ransac_iterations):
            # انتخاب تصادفی دو نقطه
            sample_idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[sample_idx]
            
            # محاسبه پارامترهای خط
            x1, y1 = p1
            x2, y2 = p2
            
            if x2 - x1 == 0:
                continue
                
            # محاسبه فاصله نقاط از خط
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            distances = np.abs(points[:,1] - (m * points[:,0] + b)) / np.sqrt(1 + m**2)
            
            # یافتن inliers
            inliers = points[distances < self.ransac_threshold]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_line = (x1, y1, x2, y2)
        
        # رسم خط بهترین
        if best_line and len(best_inliers) >= self.min_inliers:
            x1, y1, x2, y2 = best_line
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # رسم نقاط inlier
            for point in best_inliers:
                cv2.circle(result, tuple(point), 2, (0, 0, 255), -1)
        
        return result



    def curve_detection_function(self, img):
        result = img.copy()
        
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # استفاده از gaussian_kernel از سریالایزر
        kernel_size = (self.gaussian_kernel, self.gaussian_kernel)
        blur = cv2.GaussianBlur(gray, kernel_size, 0)
        
        # استفاده از آستانه‌های Canny از سریالایزر
        edges = cv2.Canny(blur, self.canny_threshold1, self.canny_threshold2)
        
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_curves = False  # برای بررسی اینکه آیا منحنی پیدا شده
        
        for contour in contours:
            if len(contour) > 15:
                try:
                    area = cv2.contourArea(contour)
                    if area > 100:
                        ellipse = cv2.fitEllipse(contour)
                        cv2.ellipse(result, ellipse, (0, 255, 0), 4)
                        cv2.drawContours(result, [contour], -1, (255, 0, 0), 2)
                        found_curves = True
                except:
                    continue
        
        # اگر هیچ منحنی پیدا نشد، لاگ بزنیم
        if not found_curves:
            logger.info("No curves detected in the image")
        
        return result




    def optimize_parameters_function(self, img):
        # ایجاد جمعیت اولیه
        population = []
        for _ in range(self.population_size):
            params = {
                'rho': np.random.uniform(self.rho_min, self.rho_max),
                'theta': np.random.uniform(self.theta_min, self.theta_max),
                'threshold': np.random.randint(30, 100),
                'minLineLength': np.random.randint(50, 200),
                'maxLineGap': np.random.randint(5, 20)
            }
            population.append(params)

        result = img.copy()
        best_params = population[0]  # انتخاب بهترین پارامترها
        
        # تبدیل به grayscale
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # اعمال پارامترهای بهینه شده
        edges = cv2.Canny(gray, 
                        best_params['threshold'], 
                        best_params['threshold'] * 2)
        
        lines = cv2.HoughLinesP(edges, 
                            best_params['rho'], 
                            best_params['theta'],
                            best_params['threshold'],
                            minLineLength=best_params['minLineLength'],
                            maxLineGap=best_params['maxLineGap'])
        
        # رسم خطوط
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return result
 




    def perspective_correction(self, img):
        result = img.copy()
        
        # پیش‌پردازش تصویر برای تشخیص بهتر پلاک
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 90, 90)
        
        # تشخیص لبه‌ها با حساسیت بالاتر
        edges = cv2.Canny(blur, 30, 150)
        
        # یافتن کانتورهای مستطیلی (احتمالاً پلاک)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # بررسی نسبت ابعاد مستطیل برای تشخیص پلاک
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w)/h
                
                # نسبت ابعاد استاندارد پلاک ایرانی
                if 2.0 < aspect_ratio < 5.0:
                    plate_contour = approx
                    
                    # تنظیم نقاط برای تبدیل پرسپکتیو
                    src_points = np.float32(plate_contour.reshape(4, 2))
                    width = int(w * 1.2)  # کمی بزرگتر برای اطمینان
                    height = int(h * 1.2)
                    
                    dst_points = np.float32([
                        [0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]
                    ])
                    
                    # اعمال تبدیل پرسپکتیو با کیفیت بالا
                    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    warped = cv2.warpPerspective(result, matrix, (width, height))
                    
                    # بهبود کیفیت نهایی
                    warped = cv2.resize(warped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    
                    # افزایش وضوح
                    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    warped = cv2.filter2D(warped, -1, kernel)
                    
                    return warped
        
        return result





    def identify_features_function(self, img1, img2):

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.SIFT_create(nfeatures=500)
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        result = cv2.drawMatches(
            img1, keypoints1,
            img2, keypoints2,
            good_matches[:50], None,
            flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )
        
        return result





    def aligned_images(self, img1, img2):
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
        
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return img1
            
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x:x.distance)[:50]
        
        if len(matches) < 4:
            return img1
            
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return img1
            
        return cv2.warpPerspective(img1, M, (width, height))


    def combine_images(self, img1, img2):
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
        
        mask = np.zeros_like(img1, dtype=np.uint8)
        center = (width // 2, height // 2)
        radius = min(width, height) // 4
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        
        img1_masked = cv2.bitwise_and(img1, mask)
        img2_masked = cv2.bitwise_and(img2, cv2.bitwise_not(mask))
        
        return cv2.add(img1_masked, img2_masked)


    def create_panorama(self, img1, img2):
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
        
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return np.hstack([img1, img2])
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            result = cv2.warpPerspective(img1, H, (width * 2, height))
            result[0:height, 0:width] = img2
            
            return result
            
        return np.hstack([img1, img2])



    def encode_image(self, img, format_choice, quality):
        """Encode processed image to bytes"""
        try:
            if format_choice == 'jpeg':
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                result, img_encoded = cv2.imencode('.jpg', img, encode_param)
            else:
                result, img_encoded = cv2.imencode(f'.{format_choice}', img)
            
            if not result:
                raise ValueError("Failed to encode image")
            
            return img_encoded
            
        except Exception as e:
            raise ValueError(f"Error encoding image: {str(e)}")

    def save_image(self, img_encoded, format_choice):
        """Save processed image to file system"""
        media_path = settings.MEDIA_ROOT
        os.makedirs(media_path, exist_ok=True)
        
        pk = self.generate_unique_id()
        file_name = f'id={pk}.{format_choice}'
        file_path = os.path.join(media_path, file_name)
        
        with open(file_path, 'wb') as f:
            f.write(img_encoded.tobytes())
        
        return pk, os.path.join(settings.MEDIA_URL, file_name)

    def generate_unique_id(self):
        """Generate unique ID for saved images"""
        media_path = settings.MEDIA_ROOT
        existing_files = os.listdir(media_path)
        return len(existing_files) + 1

    @swagger_auto_schema(
        request_body=ImageUploadSerializer,
        responses={200: 'Image optimized successfully', 400: 'Invalid image or quality'},
    )
    def post(self, request):
        try:
            serializer = ImageUploadSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            
            self.grayscale = serializer.validated_data.get('grayscale', False)
            self.denoise = serializer.validated_data.get('denoise', False)
            self.edge_detection = serializer.validated_data.get('edge_detection', False)
            self.cropping = serializer.validated_data.get('cropping')
            self.rotation_angle = serializer.validated_data.get('rotation', 0)
            self.width = serializer.validated_data.get('width')
            self.height = serializer.validated_data.get('height')
            self.gaussian_blur = serializer.validated_data.get('gaussian_blur')
            self.contrast = serializer.validated_data.get('contrast', 0)
            self.brightness = serializer.validated_data.get('brightness', 0)
            self.histogram_equalization = serializer.validated_data.get('histogram_equalization', False)
            self.corner_detection = serializer.validated_data.get('corner_detection', False)
            self.translate_x = serializer.validated_data.get('translate_x', 0)
            self.translate_y = serializer.validated_data.get('translate_y', 0)
            self.scale_x = serializer.validated_data.get('scale_x', 1.0)
            self.scale_y = serializer.validated_data.get('scale_y', 1.0)
            self.aligned_images = serializer.validated_data.get('aligned_image', False)
            self.panorama_image = serializer.validated_data.get('panorama_image', False)
            self.combine_images = serializer.validated_data.get('combine_images', False)

            self.identify_features = serializer.validated_data.get('identify_features')

            self.format_choice = serializer.validated_data.get('format_choice')
            self.quality = serializer.validated_data.get('quality')

            self.perspective_correction = serializer.validated_data.get('perspective_correction', False)

            self.kalman_line_detection = serializer.validated_data['kalman_line_detection']
            self.min_line_length = serializer.validated_data['min_line_length']
            self.max_line_gap = serializer.validated_data['max_line_gap']

            self.ransac_detection = serializer.validated_data['ransac_detection']
            self.ransac_iterations = serializer.validated_data['ransac_iterations']
            self.ransac_threshold = serializer.validated_data['ransac_threshold']
            self.min_inliers = serializer.validated_data['min_inliers']

            self.curve_detection = serializer.validated_data.get('curve_detection', False)
            self.gaussian_kernel = serializer.validated_data.get('gaussian_kernel', 5)
            self.canny_threshold1 = serializer.validated_data.get('canny_threshold1', 50)
            self.canny_threshold2 = serializer.validated_data.get('canny_threshold2', 150)


            self.optimize_parameters = serializer.validated_data.get('optimize_parameters')
            self.population_size = serializer.validated_data.get('population_size')
            self.rho_min = serializer.validated_data.get('rho_min')
            self.rho_max = serializer.validated_data.get('rho_max')
            self.theta_min = serializer.validated_data.get('theta_min')
            self.theta_max = serializer.validated_data.get('theta_max')

            self.enhance_image_quality = serializer.validated_data.get('enhance_quality', False)
            self.bilateral_d = serializer.validated_data.get('bilateral_d', 9)
            self.bilateral_sigma_color = serializer.validated_data.get('bilateral_sigma_color', 75)
            self.bilateral_sigma_space = serializer.validated_data.get('bilateral_sigma_space', 75)
            self.contrast_alpha = serializer.validated_data.get('contrast_alpha', 1.15)
            self.unsharp_weight = serializer.validated_data.get('unsharp_weight', 1.5)
            self.gaussian_sigma = serializer.validated_data.get('gaussian_sigma', 2.0)
            self.resize_factor = serializer.validated_data.get('resize_factor', 1.2)


            self.detect_numbers = serializer.validated_data.get('detect_numbers', False)



            self.img2 = None

            images = self.validate_images(
                serializer.validated_data.get('image'),
                serializer.validated_data.get('image2')
            )

            processed_img = images.get('img1')
            self.img2 = images.get('img2')
        
            if serializer.validated_data.get('panorama_image') and 'img2' in images:
                processed_img = self.create_panorama(processed_img, images['img2'])
            
            processed_img = self.process_image(processed_img)

            encoded_img = self.encode_image(
                processed_img,
                serializer.validated_data['format_choice'],
                serializer.validated_data['quality']
            )

            pk, image_url = self.save_image(encoded_img, serializer.validated_data['format_choice'])
            
            return Response({
                'message': 'Image optimized and saved',
                'image_url': image_url,
                'image_id': f'Enter your browser : http://172.105.38.184:8000/api/pk/',
                'detected_numbers': self.detected_numbers if hasattr(self, 'detected_numbers') else []

            })

        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            return Response({'error': str(e)}, status=400)




def show_image(request, pk):
    
    file_name = f'id={pk}.jpg'
    file_path = os.path.join(settings.MEDIA_ROOT, file_name)

    if not os.path.exists(file_path):
        return HttpResponse("Image not found", status=404)
    
    with open(file_path, 'rb') as image_file:
        return Response(image_file.read(), content_type="image/jpeg")
    


def image_id(request, pk):

    file_name = f'id={pk}.jpg'
    file_path = os.path.join(settings.MEDIA_ROOT, file_name)

    if not os.path.exists(file_path):
        return JsonResponse({'error': 'Image not found'}, status=404)
    
    image_url = os.path.join(settings.MEDIA_URL, file_name)

    return Response({
        'message': 'Image found',
        'image_url': image_url,
     })


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
                img = self.identify_features(img, self.img2)

            if self.aligned_images and self.img2 is not None:
                img = self.aligned_images(img, self.img2)
            
            if self.combine_images and self.img2 is not None:
                img = self.combine_images(img, self.img2)

            if self.perspective_correction:
                img = self.perspective_correction(img)

            if self.kalman_line_detection:
                logger.info("Applying Kalman line detection")
                img = self.kalman_line_detection_function(img)  # فراخوانی تابع kalman_line_detection_function
                logger.info("Kalman line detection completed")

                
            if self.ransac_detection:
                logger.info("Applying ransac line detection")
                img = self.ransac_line_detection(img)
                logger.info("ransac line detection completed")


            if self.curve_detection:
                img = self.curve_detection(img)

            if self.optimize_parameters:
                img = self.optimize_parameters(img)

            return img

        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
        
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



    def curve_detection(self, img):
        # پیش‌پردازش
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # تشخیص لبه‌ها
        edges = cv2.Canny(blur, 50, 150)
        
        # یافتن کانتورها
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # برازش منحنی
        for contour in contours:
            if len(contour) > 5:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(img, ellipse, (0,255,0), 2)

    def optimize_parameters(self):
        population = []
        for _ in range(50):
            params = {
                'rho': np.random.uniform(0.5, 2),
                'theta': np.random.uniform(np.pi/360, np.pi/90),
                'threshold': np.random.randint(30, 100),
                'minLineLength': np.random.randint(50, 200),
                'maxLineGap': np.random.randint(5, 20)
            }
            population.append(params)




    def perspective_correction(self, img):
        height, width = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        src_points = np.float32([
            [0, 0],
            [width-1, 0],
            [0, height-1],
            [width-1, height-1]
        ])
        
        dst_points = np.float32([
            [0, 0],
            [width-1, 0],
            [0, height-1],
            [width-1, height-1]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        result = cv2.warpPerspective(img, matrix, (width, height))
        
        return result



    def identify_features(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x:x.distance)[:50]  
        result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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
            self.identify_features = serializer.validated_data.get('identify_features', False)
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

            self.curve_detection = serializer.validated_data.get('curve_detections', False)
            self.optimize_parameters = serializer.validated_data.get('optimize_parameters', False)
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
                'image_id': f'Enter your browser : http://172.105.38.184:8000/api/pk/'
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


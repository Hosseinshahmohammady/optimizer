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
    parser_classes=([MultiPartParser])

    @swagger_auto_schema(
        request_body=ImageUploadSerializer,
        responses={200: 'Image optimized successfully', 400: 'Invalid image or quality'},
    )
    def post(self, request):
     try:
        serializer = ImageUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # انتقال پارامترها به کلاس
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

        images = self.validate_images(
            serializer.validated_data.get('image'),
            serializer.validated_data.get('image2')
        )

        processed_img = self.process_image(images.get('img1'))
        
        # ادامه کد...


        if serializer.validated_data.get('panorama_image') and 'img2' in images:
                processed_img = create_panorama(processed_img, images['img2'])

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
            return Response({'error': str(e)}, status=400)

def validate_images(self, image, image2):
        if not image and not image2:
            raise ValidationError('At least one image must be provided')
        
        return self.decode_images(image, image2)
    

def encode_image(self, img, format_choice, quality):
    """Encode processed image to bytes"""
    try:
        if format_choice == 'jpeg':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, img_encoded = cv2.imencode('.jpg', img, encode_param)
        elif format_choice == 'png':
            result, img_encoded = cv2.imencode('.png', img)
        elif format_choice == 'webp':
            result, img_encoded = cv2.imencode('.webp', img)
        else:
            raise ValueError(f"Unsupported format: {format_choice}")

        if not result:
            raise ValueError("Failed to encode image")

        return img_encoded

    except Exception as e:
        raise ValueError(f"Error encoding image: {str(e)}")



def decode_single_image(self, image):
        file_data = image.read()
        nparr = np.frombuffer(file_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image could not be decoded")
        return img


def generate_unique_id(self):
    """Generate unique ID for saved images"""
    media_path = settings.MEDIA_ROOT
    existing_files = os.listdir(media_path)
    return len(existing_files) + 1





def process_image(self, img):
    """Separate image processing logic"""
    try:
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
            img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Fixed RGB value

        if self.translate_x or self.translate_y:
            M = np.float32([[1, 0, self.translate_x], [0, 1, self.translate_y]])
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        if self.scale_x != 1.0 or self.scale_y != 1.0:
            img = cv2.resize(img, None, fx=self.scale_x, fy=self.scale_y, interpolation=cv2.INTER_LINEAR)

        return img

    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

    
def create_panorama(img1, img2):
    """Create panorama from two images using feature matching and homography"""
    try:
        # تبدیل تصاویر به grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # تشخیص ویژگی‌ها با ORB
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        # تطبیق ویژگی‌ها
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # مرتب‌سازی matches بر اساس فاصله
        matches = sorted(matches, key=lambda x: x.distance)
        
        # استخراج نقاط متناظر
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # محاسبه ماتریس homography
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
        
        # ایجاد تصویر پانوراما
        panorama = cv2.warpPerspective(
            img2, 
            h, 
            (img1.shape[1] + img2.shape[1], img1.shape[0])
        )
        
        # کپی تصویر اول روی پانوراما
        panorama[0:img1.shape[0], 0:img1.shape[1]] = img1
        
        return panorama

    except Exception as e:
        raise ValueError(f"Error creating panorama: {str(e)}")



def save_image(self, img_encoded, format_choice):
    """Separate file saving logic"""
    media_path = settings.MEDIA_ROOT
    os.makedirs(media_path, exist_ok=True)
    
    pk = self.generate_unique_id()
    file_name = f'id={pk}.{format_choice}'
    file_path = os.path.join(media_path, file_name)
    
    with open(file_path, 'wb') as f:
        f.write(img_encoded.tobytes())
    
    return pk, os.path.join(settings.MEDIA_URL, file_name)

























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


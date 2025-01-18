from django.contrib.auth.models import User
from django.http import JsonResponse
from django.http import HttpResponse
from django.conf import settings
from django.contrib.auth import authenticate, login
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import AllowAny
from drf_yasg.utils import swagger_auto_schema
from .seializers import ImageUploadSerializer
import os
import cv2
import base64
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
        
        return JsonResponse({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)


class OptimizeImageView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes=([MultiPartParser])

    @swagger_auto_schema(
        request_body=ImageUploadSerializer,
        responses={200: 'Image optimized successfully', 400: 'Invalid image or quality'},
    )
    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return JsonResponse({'error': 'Invalid data'}, status=status.HTTP_400_BAD_REQUEST)
     
        image = serializer.validated_data.get('image')
        image2 = serializer.validated_data.get('image2')
        if not image and not image2:
         return JsonResponse({'error': 'At least one image must be provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            if image:
                file_data = image.read()
                nparr = np.frombuffer(file_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("The image could not be decoded.")
            else:
                img = None

            if image2:
                file_data2 = image2.read()
                nparr2 = np.frombuffer(file_data2, np.uint8)
                img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
                if img2 is None:
                    raise ValueError("The second image could not be decoded.")
            else:
                img2 = None

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
                     
        format_choice = serializer.validated_data.get('format')
        quality = serializer.validated_data.get('quality')
        width = serializer.validated_data.get('width')
        height = serializer.validated_data.get('height')
        grayscale = serializer.validated_data.get('grayscale')
        denoise = serializer.validated_data.get('denoise')
        edge_detection = serializer.validated_data.get('edge_detection')
        cropping = serializer.validated_data.get('cropping')
        rotation_angle = serializer.validated_data.get('rotation')
        gaussian_blur = serializer.validated_data.get('gaussian_blur')
        histogram_equalization = serializer.validated_data.get('histogram_equalization')
        contrast = serializer.validated_data.get('contrast')
        brightness = serializer.validated_data.get('brightness')
        corner_detection = serializer.validated_data.get('corner_detection')
        Identify_features = serializer.validated_data.get('Identify_features')
        translate_x = serializer.validated_data.get('translate_x')
        translate_y = serializer.validated_data.get('translate_y')
        scale_x = serializer.validated_data.get('scale_x')
        scale_y = serializer.validated_data.get('scale_y')
        shear_x = serializer.validated_data.get('shear_x')
        shear_y = serializer.validated_data.get('shear_y')
        aligned_image = serializer.validated_data.get('aligned_image')
        combine_images = serializer.validated_data.get('combine_images')
        panorama =serializer.validated_data.get('panorama')

        try:    
            quality = int(quality)
            if quality < 1 or quality > 100:
                return JsonResponse({'error': 'Quality must be between 1 and 100'}, status=400)
        except ValueError:
                return JsonResponse({'error': 'Quality must be a valid integer'}, status=400)

        
        
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if denoise:
             img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        if edge_detection:
             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             edges = cv2.Canny(img, 100, 200)
             img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if cropping:
              try:
                  x_start, y_start, x_end, y_end = map(int, cropping.split(','))
                  img = img[y_start:y_end, x_start:x_end]
              except Exception as e:
                   return JsonResponse({'erorr': "Invalid cropping data"}, status=400)

        if rotation_angle:
             try:
                  rotation_angle = float(rotation_angle)
                  rows, cols = img.shape[ :2]
                  center = (cols / 2, rows / 2)
                  rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                  img = cv2.warpAffine(img, rotation_matrix, (cols, rows))
             except ValueError:
                  return JsonResponse({"invalid rotation angle"}, status=400)           

        if width and height:
             try:
                  width = int(width)
                  height = int(height)
                  img = cv2.resize(img, (width, height))
             except ValueError:
                  return JsonResponse({'error': 'Width and Height must be valid integers'}, status=400) 

        if gaussian_blur:
            try:
                kernel_size = int(gaussian_blur)  
                if kernel_size % 2 == 0:
                    kernel_size += 1  
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            except ValueError:
                return JsonResponse({'error': 'Gaussian blur kernel size must be a valid integer'}, status=400)
                
        if contrast or brightness:
            contrast = float(contrast) if contrast else 1.0
            brightness = int(brightness) if brightness else 0
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
            
        if histogram_equalization:
            if len(img.shape) == 3:  
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
                img = cv2.equalizeHist(img_gray)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  
            else:
                img = cv2.equalizeHist(img)

        if corner_detection:
             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             gray = np.float32(gray)
             dst = cv2.cornerHarris(gray, 2, 3, 0.04)
             dst = cv2.dilate(dst, None)
             img[dst > 0.01 * dst.max()] = [0, 0, 225]

        if Identify_features:
             gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
             sift = cv2.SIFT_create()
             keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
             keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
             bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
             matches = bf.match(descriptors1, descriptors2)
             matches = sorted(matches, key = lambda x:x.distance)
             image_matches = cv2.drawMatches(gray1, keypoints1, gray2, keypoints2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
             media_path = settings.MEDIA_ROOT
             if not os.path.exists(media_path):
              os.makedirs(media_path)

             existing_files = os.listdir(media_path)
             pk = len(existing_files) + 1
             file_name = f'features_{pk}.jpg'
             file_path = os.path.join(media_path, file_name)

             cv2.imwrite(file_path, image_matches)

             image_url = os.path.join(settings.MEDIA_URL, file_name)

             return JsonResponse({
             'message': 'Features identified and matches found',
             'image_url': image_url,
             'image_id': pk  
             })


        else:

        
         if translate_x or translate_y:
            M = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

         if scale_x != 1.0 or scale_y != 1.0:
            img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

         if shear_x or shear_y:
            M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        

         elif aligned_image:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x:x.distance)

            src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if len(img.shape) == 2:
                h, w = img.shape
            else:

             h, w, c = img.shape

             img_aligned = cv2.warpPerspective(img, M, (w, h))

            media_path = settings.MEDIA_ROOT
            if not os.path.exists(media_path):
              os.makedirs(media_path)

            existing_files = os.listdir(media_path)
            pk = len(existing_files) + 1
            file_name = f'features2_{pk}.jpg'
            file_path = os.path.join(media_path, file_name)

            cv2.imwrite(file_path, img_aligned)
            image_url = os.path.join(settings.MEDIA_URL, file_name)

            return JsonResponse({
             'message': 'Features2 identified and matches found',
             'image_url': image_url,
             'image_id': pk  
             })

         else:

            if combine_images:
                mask = np.zeros_like(img, dtype=np.uint8)
                cv2.circle(mask, (250, 250), 100, (255, 255, 255), -1)

                img_masked = cv2.bitwise_and(img, mask)
                img2_masked = cv2.bitwise_and(img2, cv2.bitwise_not(mask))

                combined = cv2.add(img_masked, img2_masked)

                media_path = settings.MEDIA_ROOT
                if not os.path.exists(media_path):
                 os.makedirs(media_path)

                existing_files = os.listdir(media_path)
                pk = len(existing_files) + 1
                file_name = f'features3_{pk}.jpg'
                file_path = os.path.join(media_path, file_name)

                cv2.imwrite(file_path, combined)
                image_url = os.path.join(settings.MEDIA_URL, file_name)

                return JsonResponse({
                'message': 'Features3 identified and matches found',
                'image_url': image_url,
                'image_id': pk  
                })

            else:
             
             if panorama:
              stitcher = cv2.Stitcher_create()
              status, panoramaa = stitcher.stitch([img, img2])

             if status == cv2.Stitcher_OK:
              cv2.imshow('Panoramaa', result)
              cv2.waitKey(0)

            media_path = settings.MEDIA_ROOT

            if not os.path.exists(media_path):
             os.makedirs(media_path)

             existing_files = os.listdir(media_path)
             pk = len(existing_files) + 1
             file_name = f'panoramaa_{pk}.jpg'
             file_path = os.path.join(media_path, file_name)

             cv2.imwrite(file_path, panoramaa)

             image_url = os.path.join(settings.MEDIA_URL, file_name)

             return JsonResponse({
             'message': 'Panorama created successfully!',
             'image_url': image_url,
             'image_id': pk
                })
            else:
        
             if format_choice == 'jpeg':
              encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
              result, img_encoded = cv2.imencode('.jpg', img, encode_param)

             elif format_choice == 'png':
              result, img_encoded = cv2.imencode('.png', img)

             elif format_choice == 'bmb':
              result, img_encoded = cv2.imencode('.bmb', img)

             elif format_choice == 'webp':
              result, img_encoded = cv2.imencode('.webp', img)

             elif format_choice == 'tiff':
              result, img_encoded = cv2.imencode('.tiff', img)

             else:
              raise ValueError("Unsupported format")
        
        if not result:
             raise ValueError("The image could not be encoded.")



        media_path = settings.MEDIA_ROOT

        if not os.path.exists(media_path):
            os.makedirs(media_path)

        existing_files = os.listdir(media_path)
        pk = len(existing_files) + 1

        file_name = f'id={pk}.{format_choice}'
        file_path = os.path.join(media_path, file_name)

        with open(file_path, 'wb') as f:
                f.write(img_encoded.tobytes())
        
        image_url = os.path.join(settings.MEDIA_URL, file_name)

        #  short_url = f"{settings.SITE_URL}/image/{pk}"

        return JsonResponse({
        'message': 'Image optimized and saved',
        'image_url': image_url,
        # 'short_url': short_url,
        'image_id': 'Enter your browser : http://172.105.38.184:8000/api/pk/'
         }) 
     


def show_image(request, pk):
    
    file_name = f'id={pk}.jpg'
    file_path = os.path.join(settings.MEDIA_ROOT, file_name)

    if not os.path.exists(file_path):
        return HttpResponse("Image not found", status=404)
    
    with open(file_path, 'rb') as image_file:
        return HttpResponse(image_file.read(), content_type="image/jpeg")
    


def image_id(request, pk):

    file_name = f'id={pk}.jpg'
    file_path = os.path.join(settings.MEDIA_ROOT, file_name)

    if not os.path.exists(file_path):
        return JsonResponse({'error': 'Image not found'}, status=404)
    
    image_url = os.path.join(settings.MEDIA_URL, file_name)

    return JsonResponse({
        'message': 'Image found',
        'image_url': image_url,
    })
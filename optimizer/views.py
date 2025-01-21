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
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from .seializers import ImageUploadSerializer
import os
import cv2
from PIL import Image
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
        serializer = ImageUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({'error': 'Invalid data'}, status=status.HTTP_400_BAD_REQUEST)
     
        
        image = serializer.validated_data.get('image')
        image2 = serializer.validated_data.get('image2')
        if not image and not image2:
         return Response({'error': 'At least one image must be provided'}, status=status.HTTP_400_BAD_REQUEST)
        
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
            return Response({'error': str(e)}, status=400)

        quality = serializer.validated_data.get('quality')
        try:
            quality = int(quality)
            if quality < 1 or quality > 100:
                return JsonResponse({'error': 'Quality must be between 1 and 100'}, status=400)
        except ValueError:
            return JsonResponse({'error': 'Quality must be a valid integer'}, status=400)



        media_path = settings.MEDIA_ROOT
        if not os.path.exists(media_path):
            os.makedirs(media_path)

        existing_files = os.listdir(media_path)
        pk = len(existing_files) + 1

        file_name = f'id={pk}.jpg'
        file_path = os.path.join(media_path, file_name)

        cv2.imwrite(file_path, img)

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


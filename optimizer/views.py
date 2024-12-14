from django.http import JsonResponse
from django.http import HttpResponse
from django.conf import settings
from django.contrib.auth import authenticate
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import AllowAny
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .seializers import ImageUploadSerializer
from PIL import Image
import os
import cv2
import numpy as np
from io import BytesIO


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
        if not image: 
            return JsonResponse({'error': 'No image exist'}, status=400)
        
        quality = request.data.get('quality')
        if quality is None:
            return JsonResponse({'error': 'Quality is required'}, status=400)
        
        img = cv2.imread(image.path)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, img_encoded = cv2.imencode('.jpg', img, encode_param)
        if not result :
            return JsonResponse({'detail':"The image could not be compressed"}, status=status.HTTP_400_BAD_REQUEST)
        


        # try:
        #     quality = int(quality)
        #     if quality < 1 or quality > 100:
        #      return JsonResponse({'error': 'Quality must be between 1 and 100'}, status=400)
        # except ValueError:
        #     return JsonResponse({'error': 'Quality must be a valid integer'}, status=400)
    
        # try:
        #     image = Image.open(image)
        # except Exception as e:
        #     return JsonResponse({'error': f'Error opening image: {str(e)}'}, status=400)
        
        # image = image.convert("RGB")

        media_path = settings.MEDIA_ROOT
        if not os.path.exists(media_path):
            os.makedirs(media_path)

        existing_files = os.listdir(media_path)
        pk = len(existing_files) + 1

        file_name = f'id={pk}.jpg'
        file_path = os.path.join(media_path, file_name)

        image.save(file_path, format='JPEG', quality=quality, optimize=True)
        
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
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.views import APIView

from rest_framework.parsers import MultiPartParser
from rest_framework.decorators import parser_classes
from drf_yasg.utils import swagger_auto_schema
from .seializers import ImageUploadSerializer

from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from PIL import Image
import io
import os
from django.conf import settings
# from .models import ImageUpload

from django.views.decorators.csrf import csrf_exempt


class OptimizeImageView(APIView):
     parser_classes=([MultiPartParser])
    #  @csrf_exempt
     @swagger_auto_schema(
    request_body=ImageUploadSerializer,
    responses={200: 'Image optimized successfully', 400: 'Invalid image or quality'}
    )
     
     def post(self, request, *args, **kwargs):
     
    #  file = request.FILES.get('image')

        serializer = ImageUploadSerializer(data=request.data) 
        if not serializer.is_valid():
                return JsonResponse({'error': 'Invalid data'}, status=status.HTTP_400_BAD_REQUEST)
     
        file = serializer.validated_data.get('image')
        # if not file: 
        #     return JsonResponse({'error': 'No image exist'}, status=400)

        quality = request.data.get('quality', 85)

        try:
            quality = int(quality)
            if quality < 1 or quality > 100:
             return JsonResponse({'error': 'Quality must be between 1 and 100'}, status=400)
        except ValueError:
            return JsonResponse({'error': 'Quality must be a valid integer'}, status=400)
    
        image = Image.open(file)

        media_path = settings.MEDIA_ROOT
        if not os.path.exists(media_path):
            os.makedirs(media_path)

        existing_files = os.listdir(media_path)
        pk = len(existing_files) + 1

        file_name = f'id={pk}.jpg'
        file_path = os.path.join(media_path, file_name)

        image.save(file_path, format='JPEG', quality=quality)

        image_url = os.path.join(settings.MEDIA_URL, file_name)
        #  short_url = f"{settings.SITE_URL}/image/{pk}"

        return JsonResponse({
        'message': 'Image optimized and saved',
        'image_url': image_url,
        # 'short_url': short_url,
        'image_id': 'Enter your browser : http://172.105.38.184:8000/api/pk/'
         }) 






# class OptimizeImageView(APIView):
#     parser_classes = [MultiPartParser]

#     @csrf_exempt  
#     @swagger_auto_schema(
#         request_body=ImageUploadSerializer,  
#         responses={200: 'Image optimized successfully', 400: 'Invalid image or quality'}
#     )
#     def post(self, request, *args, **kwargs):
        
#         serializer = ImageUploadSerializer(data=request.data)
        
#         if not serializer.is_valid():
#             return Response({"error": "Invalid data", "details": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        
       
#         image = serializer.validated_data.get('image')
#         quality = serializer.validated_data.get('quality', 85)
        
        
#         img = Image.open(image)
        
#         media_path = settings.MEDIA_ROOT
#         if not os.path.exists(media_path):
#             os.makedirs(media_path)

       
#         file_name = f"optimized_{image.name}"
#         file_path = os.path.join(media_path, file_name)
        
        
#         img.save(file_path, format='JPEG', quality=quality)

        
#         image_url = os.path.join(settings.MEDIA_URL, file_name)

#         return Response({
#             "message": "Image optimized and saved successfully",
#             "image_url": image_url,
#             'image_id': 'Enter your browser : http://172.105.38.184:8000/api/pk/'
#         }, status=status.HTTP_200_OK)






 
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
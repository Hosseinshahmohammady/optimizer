from django.http import JsonResponse
from rest_framework import status
from rest_framework.views import APIView

from rest_framework.parsers import MultiPartParser
from drf_yasg.utils import swagger_auto_schema
from .seializers import ImageUploadSerializer

from django.http import HttpResponse
from PIL import Image
import os
from django.conf import settings

class OptimizeImageView(APIView):
     parser_classes=([MultiPartParser])
     @swagger_auto_schema(
    request_body=ImageUploadSerializer,
    responses={200: 'Image optimized successfully', 400: 'Invalid image or quality'}
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

        try:
            quality = int(quality)
            if quality < 1 or quality > 100:
             return JsonResponse({'error': 'Quality must be between 1 and 100'}, status=400)
        except ValueError:
            return JsonResponse({'error': 'Quality must be a valid integer'}, status=400)
    
        try:
            image = Image.open(image)
        except Exception as e:
            return JsonResponse({'error': f'Error opening image: {str(e)}'}, status=400)
        
        image = image.convert("RGB")

        media_path = settings.MEDIA_ROOT
        if not os.path.exists(media_path):
            os.makedirs(media_path)

        existing_files = os.listdir(media_path)
        pk = len(existing_files) + 1

        file_name = f'id={pk}.jpg'
        file_path = os.path.join(media_path, file_name)

        try:
            image.save(file_path, format='JPEG', quality=quality, optimize=True)
        except Exception as e:
            return JsonResponse({'error': f'Error saving image: {str(e)}'}, status=500)

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
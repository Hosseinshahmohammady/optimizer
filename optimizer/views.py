from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from PIL import Image
import io
import os
from django.conf import settings



@api_view(['POST'])
def optimize_image(request):
     file = request.FILES.get('image')
     if not file :
          return JsonResponse({'error': 'No image provided'}, status= 400)
     
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
     short_url = f"{settings.SITE_URL}/image/{pk}"

     return JsonResponse({
        'message': 'Image optimized and saved',
        'image_url': image_url,
        'short_url': short_url
        })


     
def show_image(request, pk):
    
    file_name = f'id={pk}.jpg'
    file_path = os.path.join(settings.MEDIA_ROOT, file_name)

    if not os.path.exists(file_path):
        return HttpResponse("Image not found", status=404)
    
    with open(file_path, 'rb') as image_file:
        return HttpResponse(image_file.read(), content_type="image/jpeg")
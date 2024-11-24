from django.shortcuts import render

from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser
from PIL import Image
import io
import os
from django.conf import settings



@api_view(['POST'])
def optimize_image(request):
     file = request.FILES.get('image')
     if not file :
          return JsonResponse({'error': 'No image provided'}, status= 400)
    
     image = Image.open(file)


     media_path = settings.MEDIA_ROOT
     if not os.path.exists(media_path):
        os.makedirs(media_path)


     file_name = 'optimized_image.jpg'
     file_path = os.path.join(media_path, file_name)

     image.save(file_path, format='JPEG', quality=85)

     image_url = os.path.join(settings.MEDIA_URL, file_name)


     return JsonResponse({
        'message': 'Image optimized and saved',
        'image_url': image_url
        })


     # output = io.BytesIO()
     # image.save(output, format='JPEG', quality = 85)
     # output.seek(0)

     # return JsonResponse({
        
     #         'message': 'image optimized',
     #          'image': output.getvalue().hex()
              
     #          })
from django.shortcuts import render

from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser
from PIL import Image
import io


@api_view(['POST'])
def optimize_image(request):
    file = request.FILES.get('image')
    if not file :
         return JsonResponse({'error': 'No image provided'}, status= 400)
    
    image = Image.open(file)

    output = io.BytesIO()
    image.save(output, format='JPEG', quality = 85)
    output.seek(0)


    return JsonResponse(
         {
              'message': 'image optimized',
              'image': output.getvalue().hex()
         }
    )
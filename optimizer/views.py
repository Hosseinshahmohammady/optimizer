from django.contrib.sites.shortcuts import get_current_site
from django.shortcuts import render, redirect, get_object_or_404, HttpResponseRedirect
from django.utils.encoding import force_text
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.http import HttpResponse
from django.conf import settings
from django.contrib.auth import authenticate, login
from django.core.files.storage import default_storage
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_decode
from django.utils.encoding import force_bytes
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import AllowAny
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .seializers import ImageUploadSerializer
import os
import cv2
import numpy as np
from io import BytesIO
from .tokens import account_activation_token
from .forms import SignUpForm
from .tokens import account_activation_token

def home_view(request):
    return render(request, 'home.html')

def activation_sent_view(request):
    return render(request, 'activation_sent.html')


def activate(request, uidb64, token):
    try:
        uid = force_text(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.profile.signup_confirmation = True
        user.save()
        login(request, user)
        return redirect('home')
    else:
        return render(request, 'activation_invalid.html')

def signup_view(request):
    if request.method  == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
               user = form.save()
               user.refresh_from.db()
               user.profile.first_name = form.cleaned_data.get('first_name')
               user.profile.last_name = form.cleaned_data.get('last_name')
               user.profile.email = form.cleaned_data.get('email')

               user.is_active = False
               user.save()
               current_site = get_current_site(request)
               subject = 'Please Activate Your Account'

               message = render_to_string('activation_request.html', {
                'user': user,
                'domain': current_site.domain,
                'uid': urlsafe_base64_decode(force_bytes(user.pk)),
                'token': account_activation_token.make_token(user),
            })
        user.email_user(subject, message)
        return redirect('activation_sent')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})

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

        try:    
            quality = int(quality)
            if quality < 1 or quality > 100:
                return JsonResponse({'error': 'Quality must be between 1 and 100'}, status=400)
        except ValueError:
                return JsonResponse({'error': 'Quality must be a valid integer'}, status=400)

        
        try:
            file_data = image.read()
            nparr = np.frombuffer(file_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("The image could not be decoded.")
        except Exception as e:    
                return JsonResponse({'error': str(e)}, status=400)

            
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, img_encoded = cv2.imencode('.jpg', img, encode_param)
        if not result:
            raise ValueError("The image could not be compressed.")
    
        
        media_path = settings.MEDIA_ROOT
        if not os.path.exists(media_path):
            os.makedirs(media_path)

        existing_files = os.listdir(media_path)
        pk = len(existing_files) + 1

        file_name = f'id={pk}.jpg'
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
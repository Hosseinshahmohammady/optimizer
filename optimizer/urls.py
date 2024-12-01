from django.urls import path
from .views import optimize_image
from . import views


urlpatterns = [
path('optimize/', optimize_image, name='optimize_image'),
path('show_image/<int:pk>/', views.show_image, name='show_image'),
# path('image/<int:pk>/', views.show_image, name='show_image'),
path('get-image-url/<int:pk>/', views.get_image_url_by_id, name='get_image_url_by_id'),


]
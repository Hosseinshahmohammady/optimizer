from django.urls import path
from .views import optimize_image


urlpatterns = [
    path('optimize/', optimize_image, name='optimize_image')
]
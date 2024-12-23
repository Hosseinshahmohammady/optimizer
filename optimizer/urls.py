from django.urls import path
from .views import OptimizeImageView, signup_view, home_view, login_view
from .import views

from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from django.http import HttpResponse
from rest_framework import routers
from rest_framework_simplejwt import views as jwt_views
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated



schema_view = get_schema_view(
    openapi.Info(
        title="Image Optimizer API",
        default_version='v1',
        description="API for optimizing and reducing image size",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@yourdomain.com"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=[AllowAny],
    authentication_classes=[],
)

urlpatterns = [
path('optimize/', OptimizeImageView.as_view(), name='optimize_image'),
path('show_image/<int:pk>/', views.show_image, name='show_image'),
# path('image/<int:pk>/', views.show_image, name='show_image'),
path('<int:pk>/', views.image_id, name='url_id'),
path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
path('swagger.yaml', schema_view.without_ui(cache_timeout=0), name='swagger-yaml'),
path('api/token/', jwt_views.TokenObtainPairView.as_view(), name='token_obtain_pair'),
path('api/token/refresh/', jwt_views.TokenRefreshView.as_view(), name='token_refresh'),
path('home_optimize/', home_view, name='home_optimize'),
path('signup/', signup_view, name="signup"),
path('login/', login_view, name='login')
]
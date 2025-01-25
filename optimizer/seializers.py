from rest_framework import serializers
import numpy as np

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    image2 = serializers.ImageField(required=False)

    format_choice = serializers.ChoiceField(
        choices=['jpeg', 'png', 'bmb', 'webp', 'tiff'],
          default='jpeg'
          )


    quality = serializers.IntegerField(
        required=False,
          min_value=1,
            max_value=100,
              default=95
              )

        # پارامترهای پردازش تصویر
    grayscale = serializers.BooleanField(required=False, default=False)
    denoise = serializers.BooleanField(required=False, default=False)
    edge_detection = serializers.BooleanField(required=False, default=False)
    corner_detection = serializers.BooleanField(required=False, default=False)

        # پارامترهای تغییر اندازه و چرخش
    width = serializers.IntegerField(required=False, min_value=1)
    height = serializers.IntegerField(required=False, min_value=1)
    rotation = serializers.IntegerField(required=False, default=0)

        # پارامترهای فیلتر و بهبود
    gaussian_blur = serializers.BooleanField(required=False, default=False)
    contrast = serializers.FloatField(required=False, min_value=-100, max_value=100, default=0)  
    brightness = serializers.FloatField(required=False, min_value=-100, max_value=100, default=0)
    histogram_equalization = serializers.BooleanField(required=False, default=False)

        # پارامترهای برش و تغییر مقیاس
    cropping = serializers.CharField(required=False, help_text="Format: 'x_start,y_start,x_end,y_end'")
    translate_x = serializers.IntegerField(required=False, default=0)
    translate_y = serializers.IntegerField(required=False, default=0)
    scale_x = serializers.FloatField(required=False, default=1.0)
    scale_y = serializers.FloatField(required=False, default=1.0)
    shear_x = serializers.FloatField(required=False, default=0)
    shear_y = serializers.FloatField(required=False, default=0)

        # پارامترهای ترکیب تصاویر
    # Identify_features = serializers.BooleanField(required=False, default=False)
    # aligne_image = serializers.BooleanField(required=False, default=False)
    # combine_images = serializers.BooleanField(required=False, default=False)
    # panorama_image = serializers.BooleanField(required=False, default=False)
    perspective = serializers.BooleanField(required=False, default=False)

    # اضافه کردن پارامترهای تشخیص خط کالمن
    kalman_line_detection = serializers.BooleanField(default=False)
    min_line_length = serializers.IntegerField(default=100, min_value=10, required=False)
    max_line_gap = serializers.IntegerField(default=10, min_value=1, required=False)
    
    # پارامترهای RANSAC
    ransac_detection = serializers.BooleanField(default=False)
    ransac_iterations = serializers.IntegerField(default=100, min_value=10, required=False)
    ransac_threshold = serializers.FloatField(default=3.0, min_value=0.1, required=False)
    
    # پارامترهای تشخیص منحنی
    curve_detection = serializers.BooleanField(default=False)
    gaussian_kernel = serializers.IntegerField(default=5, min_value=3, required=False)
    canny_threshold1 = serializers.IntegerField(default=50, min_value=1, required=False)
    canny_threshold2 = serializers.IntegerField(default=150, min_value=1, required=False)
    
    # پارامترهای بهینه‌سازی
    optimize_parameters = serializers.BooleanField(default=False)
    population_size = serializers.IntegerField(default=50, min_value=10, required=False)
    rho_min = serializers.FloatField(default=0.5, required=False)
    rho_max = serializers.FloatField(default=2.0, required=False)
    theta_min = serializers.FloatField(default=np.pi/360, required=False)
    theta_max = serializers.FloatField(default=np.pi/90, required=False)

    def validate_quality(self, value):
        if value is None:
            raise serializers.ValidationError("Quality is required")
        if value < 1 or value > 100:
            raise serializers.ValidationError("Quality must be between 1 and 100")
        return value

    def validate_cropping(self, value):
        if value:
            try:
                x_start, y_start, x_end, y_end = map(int, value.split(','))
                if x_start < 0 or y_start < 0 or x_end < x_start or y_end < y_start:
                    raise serializers.ValidationError("Invalid cropping coordinates")
            except ValueError:
                raise serializers.ValidationError("Cropping format should be 'x_start,y_start,x_end,y_end'")
        return value

    def validate(self, data):
        if data.get('combine_images') or data.get('panorama_image'):
            if not data.get('image2'):
                raise serializers.ValidationError("Second image required for selected operations")
        return data

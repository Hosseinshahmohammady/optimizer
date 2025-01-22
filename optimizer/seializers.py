from rest_framework import serializers

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    image2 = serializers.ImageField(required=False)
    quality = serializers.IntegerField(required=False, min_value=1, max_value=100, default=95)
    width = serializers.IntegerField(required=False, min_value=1)
    height = serializers.IntegerField(required=False, min_value=1)
    grayscale = serializers.BooleanField(required=False, default=False)
    denoise = serializers.BooleanField(required=False, default=False)
    edge_detection = serializers.BooleanField(required=False, default=False)
    format_choice = serializers.ChoiceField(choices=['jpeg', 'png', 'bmb', 'webp', 'tiff'], default='jpeg')
    cropping = serializers.CharField(required=False, help_text="Format: 'x_start,y_start,x_end,y_end'")
    rotation = serializers.IntegerField(required=False, default=0)
    gaussian_blur = serializers.BooleanField(required=False, default=False)
    contrast = serializers.FloatField(required=False, min_value=-100, max_value=100, default=0)  
    brightness = serializers.FloatField(required=False, min_value=-100, max_value=100, default=0)
    corner_detection = serializers.BooleanField(required=False, default=False)
    translate_x = serializers.IntegerField(required=False, default=0)
    translate_y = serializers.IntegerField(required=False, default=0)
    scale_x = serializers.FloatField(required=False, default=1.0)
    scale_y = serializers.FloatField(required=False, default=1.0)
    shear_x = serializers.FloatField(required=False, default=0)
    shear_y = serializers.FloatField(required=False, default=0)
    histogram_equalization = serializers.BooleanField(required=False, default=False)
    Identify_features = serializers.BooleanField(required=False, default=False)
    aligned_image = serializers.BooleanField(required=False, default=False)
    combine_images = serializers.BooleanField(required=False, default=False)
    panorama_image = serializers.BooleanField(required=False, default=False)


    def validate_quality(self, value):
        if value is None:
            raise serializers.ValidationError("Quality is required")
        if value < 1 or value > 100:
            raise serializers.ValidationError("Quality must be between 1 and 100")
        return value


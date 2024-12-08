# from rest_framework import serializers

# class ImageUploadSerializer(serializers.Serializer):
#     image = serializers.ImageField()
#     quality = serializers.IntegerField(
#         required=False, 
#         min_value=1, 
#         max_value=100, 
#         default=85
#     )


from rest_framework import serializers
from .models import ImageUpload

class ImageUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageUpload
        fields = ['image', 'quality']

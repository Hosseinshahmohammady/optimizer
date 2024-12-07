from rest_framework import serializers

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    quality = serializers.IntegerField(required=False, default=85)

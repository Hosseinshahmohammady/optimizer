from rest_framework import serializers

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    quality = serializers.IntegerField( 
        min_value=1, 
        max_value=100,
        default=85
    )


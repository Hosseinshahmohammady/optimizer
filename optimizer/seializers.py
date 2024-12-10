from rest_framework import serializers

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    quality = serializers.IntegerField(
        required=True, 
        min_value=1, 
        max_value=100, 
        default=85
    )

    def validate_quality(self, value):
        if value < 1 or value > 100 :
            raise
        serializers.ValidationError("Quality must be betweet 1 and 100.")
        return value
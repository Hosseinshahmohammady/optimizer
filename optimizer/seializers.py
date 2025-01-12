from rest_framework import serializers

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    quality = serializers.IntegerField(min_value=1, max_value=100, default=85)
    width = serializers.IntegerField(required=False, min_value=1)
    height = serializers.IntegerField(required=False, min_value=1)
    grayscale = serializers.BooleanField(required=False, default=False)
    denoise = serializers.BooleanField(required=False, default=False)
    edge_detection = serializers.BooleanField(required=False, default=False)

    def validate_quality(self, value):
        if value is None:
            raise serializers.ValidationError("Quality is required")
        if value < 1 or value > 100:
            raise serializers.ValidationError("Quality must be between 1 and 100")
        return value


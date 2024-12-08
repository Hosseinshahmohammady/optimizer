from django.db import models

class ImageUpload(models.Model):
    image = models.ImageField(upload_to='images/')
    quality = models.IntegerField(default=100)  # Quality field, 0-100 range

    def __str__(self):
        return f"Image {self.id} with quality {self.quality}"

from django.contrib import admin

from .models import ImageUpload


@admin.register(ImageUpload)
class ImageUpload(admin.ModelAdmin):
    pass


# admin.site.register(ImageUpload, ImageUploadAdmin)
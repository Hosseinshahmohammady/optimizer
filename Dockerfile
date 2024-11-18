# استفاده از تصویر پایه Python
FROM python:3.9-slim

# تنظیم پوشه کاری
WORKDIR /app

# کپی کردن فایل requirements.txt به داخل کانتینر
COPY requirements.txt /app/

# نصب وابستگی‌ها
RUN pip install --no-cache-dir -r requirements.txt

# کپی کردن تمام فایل‌های پروژه Django به داخل کانتینر
COPY . /app/

# اجرای migration‌ها
RUN python manage.py migrate

# پورت 8000 را برای دسترسی به پروژه باز می‌کنیم
EXPOSE 8000

# دستور اجرا برای راه‌اندازی Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

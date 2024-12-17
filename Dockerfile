FROM python:3.10-slim-buster

LABEL maintainer="hoseinshahmohammady@gmail.com"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# تغییر منابع APT به سرور دیگری
RUN sed -i 's/http:\/\/deb.debian.org/http:\/\/deb.debian.org\/debian/g' /etc/apt/sources.list && \
    sed -i 's/http:\/\/security.debian.org/http:\/\/deb.debian.org\/debian-security/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y \
    gnupg \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    wait-for-it && \
    apt-get clean

# نصب pip و بسته‌های پایتون از requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نصب opencv-python
RUN pip install --no-cache-dir opencv-python

# کپی کردن کدهای برنامه
COPY . /app/

# expose پورت 8000
EXPOSE 8000

# اجرای دستور wait-for-it برای اطمینان از آماده بودن پایگاه داده
CMD ["sh", "-c", "/app/wait-for-it.sh db:5432 -- python manage.py runserver 0.0.0.0:8000"]




# FROM python:3.10-slim-buster

# LABEL maintainer="hoseinshahmohammady@gmail.com"

# ENV PYTHONUNBUFFERED=1

# WORKDIR /app

# RUN sed -i 's/http:\/\/deb.debian.org/http:\/\/ftp.debian.org/g' /etc/apt/sources.list && \
#     apt-get update && \
#     apt-get install -y \
#     gnupg \
#     curl \
#     libglib2.0-0 \
#     libsm6 \
#     libxrender1 \
#     libxext6 \
#     libgl1-mesa-glx \
#     wait-for-it && \
#     apt-get clean

#     COPY requirements.txt .
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# RUN pip install --no-cache-dir opencv-python

# COPY . /app/

# EXPOSE 8000

# # RUN apt-get update && apt-get install -y wait-for-it

# CMD ["sh", "-c", "/app/wait-for-it.sh db:5432 -- python manage.py runserver 0.0.0.0:8000"]



# RUN pip install wait-for-it

# نصب wait-for-it
# RUN apt-get update && apt-get install -y wait-for-it


# COPY requirements.txt .

# RUN apt-get update && apt-get install -y libpq-dev

# RUN pip install --no-cache-dir -r requirements.txt

# RUN pip install --upgrade pip && pip3 install -r requirements.txt

# RUN apt-get update && apt-get install -y gnupg

# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138

# RUN sed -i 's/http:\/\/deb.debian.org/http:\/\/ftp.debian.org/g' /etc/apt/sources.list

# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 \
#     libsm6 \
#     libxrender1 \
#     libxext6 \
#     libgl1-mesa-glx \
#     && apt-get clean

# RUN pip install --no-cache-dir opencv-python

# RUN apt-get update && apt-get install -y \
#     libpq-dev \
#     build-essential

# COPY . /app/

# RUN python manage.py migrate

# EXPOSE 8000

# RUN apt-get update && apt-get install -y wait-for-it

# CMD ["sh", "-c", "/app/wait-for-it.sh db:5432 -- python manage.py runserver 0.0.0.0:8000"]
FROM python:3.10-slim-buster

LABEL maintainer="hoseinshahmohammady@gmail.com"

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev

COPY requirements.txt .

RUN pip install --upgrade pip && pip3 install -r requirements.txt

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean

RUN pip install --no-cache-dir opencv-python

COPY . /app/


EXPOSE 8000


CMD ["sh", "-c", "/app/wait-for-it.sh db:5432 -- python manage.py runserver 0.0.0.0:8000"]
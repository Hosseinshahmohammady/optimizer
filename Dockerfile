FROM python:3.10-slim-buster

LABEL maintainer="hoseinshahmohammady@gmail.com"

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# RUN pip install wait-for-it

# نصب wait-for-it
# RUN apt-get update && apt-get install -y wait-for-it


COPY requirements.txt .

# RUN apt-get update && apt-get install -y libpq-dev

# RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade pip && pip3 install -r requirements.txt

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean

RUN pip install --no-cache-dir opencv-python

# RUN apt-get update && apt-get install -y \
#     libpq-dev \
#     build-essential

COPY . /app/

# RUN python manage.py migrate

EXPOSE 8000

RUN apt-get update && apt-get install -y wait-for-it

CMD ["sh", "-c", "/app/wait-for-it.sh db:5432 -- python manage.py runserver 0.0.0.0:8000"]
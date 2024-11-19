FROM python:3.10-slim-buster

LABEL maintainer="hoseinshahmohammady@gmail.com"

ENV PYTHONUNBUFFERED=1

WORKDIR /src/usr/app

# RUN pip install wait-for-it

# نصب wait-for-it
# RUN apt-get update && apt-get install -y wait-for-it


COPY requirements.txt .

# RUN apt-get update && apt-get install -y libpq-dev

# RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade pip && pip3 install -r requirements.txt

# RUN apt-get update && apt-get install -y \
#     libpq-dev \
#     build-essential

COPY . /app/

# RUN python manage.py migrate

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

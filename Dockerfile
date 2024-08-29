FROM python:3.10

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

USER root

RUN mkdir -p /app/Crafty/outputs && chmod -R 777 /app/Crafty/outputs

RUN pip install --no-cache-dir -r Crafty/requirements.txt

WORKDIR /app/Crafty


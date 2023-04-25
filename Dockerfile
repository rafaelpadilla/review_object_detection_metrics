FROM python:3.9.16
COPY requirements.txt /requirements.txt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    pip3 install --upgrade pip && \
    pip3 install -r /requirements.txt

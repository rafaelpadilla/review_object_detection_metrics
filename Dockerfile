FROM python:3.9.16
ENV DISPLAY = ${DISPLAY}
RUN echo "export DISPLAY=:${DISPLAY}" >> /etc/profile
COPY requirements.txt /requirements.txt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install libgl1-mesa-dev libosmesa6-dev \
    xvfb patchelf ffmpeg libsm6 libxext6 x11-xserver-utils -y && \
    pip3 install --upgrade pip && \
    pip3 install -r /requirements.txt

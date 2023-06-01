FROM python:3.10.6
ENV DISPLAY = ${DISPLAY}
RUN echo "export DISPLAY=:${DISPLAY}" >> /etc/profile
COPY requirements.txt /requirements.txt
COPY src/app.py /app.py
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y  python3-pyqt5
ENTRYPOINT QT_DEBUG_PLUGINS=1 python /app.py


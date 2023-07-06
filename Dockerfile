FROM jozo/pyqt5
ENV DISPLAY = ${DISPLAY}
RUN echo "export DISPLAY=:${DISPLAY}" >> /etc/profile
COPY requirements.txt /requirements.txt

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN apt-get upgrade -y && \
    apt-get install -y python3-pip && \
    pip3 install -r /requirements.txt
COPY . /review_object_detection_metrics
RUN python3 /review_object_detection_metrics/setup.py install
ENTRYPOINT QT_DEBUG_PLUGINS=1 python3 review_object_detection_metrics/run.py
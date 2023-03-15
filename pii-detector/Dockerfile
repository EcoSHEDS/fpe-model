FROM pytorch/torchserve:0.5.3-cpu

COPY ./deployment/config.properties /home/model-server/config.properties
COPY ./lib /home/model-server/lib

ENV PYTHONPATH=/home/model-server/lib/yolov5:/home/model-server/lib/cameratraps

USER root
# RUN pip install --upgrade pip && pip install wheel
RUN pip install opencv-python jsonpickle
USER model-server

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

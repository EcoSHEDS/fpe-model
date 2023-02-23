FROM pytorch/torchserve:0.5.3-cpu

# RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY ./deployment/config.properties /home/model-server/config.properties

# WORKDIR /home/model-server
# ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

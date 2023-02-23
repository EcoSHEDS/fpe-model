FROM pytorch/torchserve:0.5.3-cpu

COPY ./deployment/config.properties /home/model-server/config.properties

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

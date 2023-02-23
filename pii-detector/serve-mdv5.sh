#!/bin/bash

torchserve --start --model-store /app/model_store --no-config-snapshots --models mdv5=/app/mdv5.mar

# example: curl http://127.0.0.1:8080/predictions/mdv5 -T path/to/image.jpg

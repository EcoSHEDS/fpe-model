#!/bin/bash
# Serve mdv5 model with TorchServe

torchserve --start --model-store /app/models/mdv5 --no-config-snapshots --models mdv5a=/app/mdv5a.mar
# example: curl http://127.0.0.1:8080/predictions/mdv5a -T path/to/image.jpg

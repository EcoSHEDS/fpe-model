#!/bin/bash
# Run TorchServe in a Docker container

docker run --rm -p 8080:8080 -p 8081:8081 -p 8082:8082 -v "$(pwd)":/app -it pytorch/torchserve:0.5.3-cpu bash /app/serve-mdv5.sh

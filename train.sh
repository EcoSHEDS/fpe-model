#!/bin/bash

# Default values
EXPERIMENT_NAME="fpe-rank"
DATA_DIR="/home/jeff/data/fpe/WESTB0/models/RANK-FLOW-20240424/input"
IMAGES_DIR="/home/jeff/data/fpe/images"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --images-dir)
            IMAGES_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found: $IMAGES_DIR"
    exit 1
fi

# Run the training container
docker run -it \
    --runtime=nvidia \
    --gpus all \
    -v "${DATA_DIR}:/opt/ml/input/data/data" \
    -v "${IMAGES_DIR}:/opt/ml/input/data/images" \
    -v /opt/ml:/opt/ml \
    -v "$(pwd):/app" \
    -w /app \
    --env-file .env.docker \
    --network host \
    --shm-size=4g \
    fpe-rank \
    python src/train.py \
        --mlflow-experiment-name "${EXPERIMENT_NAME}"

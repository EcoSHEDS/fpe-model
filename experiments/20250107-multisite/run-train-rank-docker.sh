#!/bin/bash

PRETRAINED=false
SKIP_TRAIN=false
RUN=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pretrained)
            PRETRAINED=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --run)
            RUN="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --run <run_number> [--pretrained] [--skip-train]"
            exit 1
            ;;
    esac
done

if [ -z "$RUN" ]; then
    echo "Error: Run number is required"
    echo "Usage: $0 --run <run_number> [--pretrained] [--skip-train]"
    exit 1
fi

# Default values
IMAGES_DIR="/home/jeff/data/fpe/images"
DATA_DIR="/home/jeff/git/fpe-model/experiments/20250107-multisite/runs/${RUN}/input"
OUTPUT_DIR="/home/jeff/git/fpe-model/experiments/20250107-multisite/runs/${RUN}/output"
MODEL_DIR="${OUTPUT_DIR}/model"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found: $IMAGES_DIR"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Build the command arguments
CMD_ARGS="python train-rank.py"

if [ "$PRETRAINED" = true ]; then
    CMD_ARGS="$CMD_ARGS --pretrained-model-path /opt/ml/input/data/data/model.pth --load-optimizer-state"
fi

if [ "$SKIP_TRAIN" = true ]; then
    CMD_ARGS="$CMD_ARGS --skip-train"
fi

docker run \
    --runtime=nvidia \
    --gpus all \
    -v "${DATA_DIR}:/opt/ml/input/data/data" \
    -v "${IMAGES_DIR}:/opt/ml/input/data/images" \
    -v "${MODEL_DIR}:/opt/ml/model" \
    -v "${OUTPUT_DIR}:/opt/ml/output" \
    -v /opt/ml:/opt/ml \
    -v "$(pwd):/app" \
    -w /app \
    --env-file .env.docker \
    --network host \
    --shm-size=4g \
    fpe-rank \
    $CMD_ARGS

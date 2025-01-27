#!/bin/bash

MODEL=""
STATION_ID=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    echo "Processing argument: ${1} ${2}"
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --station-id)
            STATION_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model> --station-id <station_id>"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Error: Model is required"
    echo "Usage: $0 --model <model> --station-id <station_id>"
    exit 1
fi

if [ -z "$STATION_ID" ]; then
    echo "Error: Station ID is required"
    echo "Usage: $0 --model <model> --station-id <station_id>"
    exit 1
fi

# Default values
IMAGES_DIR="$(pwd)/images"
DATA_DIR="$(pwd)/runs/${MODEL}/${STATION_ID}/input"
OUTPUT_DIR="$(pwd)/runs/${MODEL}/${STATION_ID}/output"
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
    mkdir -p "$OUTPUT_DIR"
fi

if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p "$MODEL_DIR"
fi

# Build the command arguments
CMD_ARGS="python src/ranknet_aux.py"

echo "Data directory: $DATA_DIR"
echo "Images directory: $IMAGES_DIR"
echo "Model directory: $MODEL_DIR"
echo "Output directory: $OUTPUT_DIR"

echo "Running command: $CMD_ARGS"

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
    --env-file "$(pwd)/../../.env.docker" \
    --network host \
    --shm-size=4g \
    fpe-rank \
    $CMD_ARGS


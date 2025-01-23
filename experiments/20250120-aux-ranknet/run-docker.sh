#!/bin/bash

EXPERIMENT="20250120-aux-ranknet"
RUN=""
STATION=""
MODEL=""
AUX_MODEL=""
AUX_FILE=""
AUX_LSTM_TIMESTEP=""
SEED="1610"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    echo "Processing argument: ${1} ${2}"
    case $1 in
        --run)
            RUN="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --station-id)
            STATION="$2"
            shift 2
            ;;
        --aux-model)
            AUX_MODEL="$2"
            shift 2
            ;;
        --aux-file)
            AUX_FILE="$2"
            shift 2
            ;;
        --aux-lstm-timestep)
            AUX_LSTM_TIMESTEP="${2}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --run <run_name> --model <model_script> --station-id <station_id> --aux-model <aux_model> --aux-file <aux_file> --aux-lstm-timestep <timestep>"
            exit 1
            ;;
    esac
done

if [ -z "$RUN" ]; then
    echo "Error: Run number is required"
    echo "Usage: $0 --run <run_name> --model <model_script> --station-id <station_id> --aux-model <aux_model> --aux-file <aux_file> --aux-lstm-timestep <timestep>"
    exit 1
fi

if [ -z "$STATION" ]; then
    echo "Error: Station ID is required"
    echo "Usage: $0 --run <run_name> --model <model_script> --station-id <station_id> --aux-model <aux_model> --aux-file <aux_file> --aux-lstm-timestep <timestep>  "
    exit 1
fi

if [ -z "$MODEL" ]; then
    echo "Error: Model script is required"
    echo "Usage: $0 --run <run_name> --model <model_script> --station-id <station_id> --aux-model <aux_model> --aux-file <aux_file> --aux-lstm-timestep <timestep>"
    exit 1
fi

# Default values
IMAGES_DIR="$(pwd)/images"
DATA_DIR="$(pwd)/runs/${RUN}/stations/${STATION}/input"
OUTPUT_DIR="$(pwd)/runs/${RUN}/stations/${STATION}/output"
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
CMD_ARGS="python models/$MODEL.py --mlflow-experiment $EXPERIMENT --mlflow-run $RUN-$STATION --random-seed $SEED"

if [ -n "$AUX_FILE" ]; then
    CMD_ARGS="$CMD_ARGS --aux-file $AUX_FILE"
fi

if [ -n "$AUX_MODEL" ]; then
    CMD_ARGS="$CMD_ARGS --aux-model $AUX_MODEL"
fi

if [ -n "$AUX_LSTM_TIMESTEP" ]; then
    CMD_ARGS="$CMD_ARGS --aux-lstm-timestep ${AUX_LSTM_TIMESTEP}"
fi

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


#!/bin/bash

PRETRAINED=false
SKIP_TRAIN=false
RUN=""
STATION_ID=""
STATIONS_FILE=""
LEARNING_RATE=""
EARLY_STOPPING_PATIENCE=""

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
        --station-id)
            STATION_ID="$2"
            shift 2
            ;;
        --stations-file)
            STATIONS_FILE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --early-stopping-patience)
            EARLY_STOPPING_PATIENCE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --run <run_number> (--station-id <station_id> | --stations-file <file>) [--pretrained] [--skip-train] [--lr <learning_rate>] [--early-stopping-patience <patience>]"
            exit 1
            ;;
    esac
done

if [ -z "$RUN" ]; then
    echo "Error: Run number is required"
    echo "Usage: $0 --run <run_number> (--station-id <station_id> | --stations-file <file>) [--pretrained] [--skip-train] [--lr <learning_rate>] [--early-stopping-patience <patience>]"
    exit 1
fi

if [ -z "$STATION_ID" ] && [ -z "$STATIONS_FILE" ]; then
    echo "Error: Either station ID or stations file is required"
    echo "Usage: $0 --run <run_number> (--station-id <station_id> | --stations-file <file>) [--pretrained] [--skip-train] [--lr <learning_rate>] [--early-stopping-patience <patience>]"
    exit 1
fi

if [ ! -z "$STATION_ID" ] && [ ! -z "$STATIONS_FILE" ]; then
    echo "Error: Cannot specify both station ID and stations file"
    echo "Usage: $0 --run <run_number> (--station-id <station_id> | --stations-file <file>) [--pretrained] [--skip-train] [--lr <learning_rate>] [--early-stopping-patience <patience>]"
    exit 1
fi

# Function to run docker command for a single station
run_station() {
    local station="$1"
    echo "Processing station: $station"

    # Default values
    IMAGES_DIR="/home/jeff/data/fpe/images"
    DATA_DIR="/home/jeff/git/fpe-model/experiments/20250107-multisite/runs/run-${RUN}/station-${station}/input"
    OUTPUT_DIR="/home/jeff/git/fpe-model/experiments/20250107-multisite/runs/run-${RUN}/station-${station}/output"
    MODEL_DIR="${OUTPUT_DIR}/model"

    if [ ! -d "$DATA_DIR" ]; then
        echo "Error: Data directory not found: $DATA_DIR"
        return 1
    fi

    if [ ! -d "$IMAGES_DIR" ]; then
        echo "Error: Images directory not found: $IMAGES_DIR"
        return 1
    fi

    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Error: Output directory not found: $OUTPUT_DIR"
        return 1
    fi

    if [ ! -d "$MODEL_DIR" ]; then
        echo "Error: Model directory not found: $MODEL_DIR"
        return 1
    fi

    # Build the command arguments
    CMD_ARGS="python train-rank.py"

    if [ "$PRETRAINED" = true ]; then
        CMD_ARGS="$CMD_ARGS --pretrained-model-path /opt/ml/input/data/data/model.pth --load-optimizer-state"
    fi

    if [ "$SKIP_TRAIN" = true ]; then
        CMD_ARGS="$CMD_ARGS --skip-train"
    fi

    if [ ! -z "$LEARNING_RATE" ]; then
        CMD_ARGS="$CMD_ARGS --lr $LEARNING_RATE"
    fi

    if [ ! -z "$EARLY_STOPPING_PATIENCE" ]; then
        CMD_ARGS="$CMD_ARGS --early-stopping-patience $EARLY_STOPPING_PATIENCE"
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
}

# Process either single station or multiple stations from file
if [ ! -z "$STATION_ID" ]; then
    # Single station
    run_station "$STATION_ID"
else
    if [ ! -f "$STATIONS_FILE" ]; then
        echo "Error: Stations file not found: $STATIONS_FILE"
        exit 1
    fi

    while IFS= read -r station || [ -n "$station" ]; do
        # Skip empty lines and comments
        if [ -z "$station" ] || [[ "$station" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        run_station "$station"
    done < "$STATIONS_FILE"
fi

#!/bin/bash
# Print help message
print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Run a model training experiment"
    echo
    echo "Options:"
    echo "  --experiments-dir DIR    Base directory containing all experiments (default: /home/jeff/data/fpe/experiments)"
    echo "  --experiment-name NAME   Name of the experiment (default: test-experiment)"
    echo "  --run-name NAME         Name of the specific run (default: test-run)"
    echo "  --images-dir DIR        Directory containing image files (default: /home/jeff/data/fpe/images)"
    echo "  -h, --help             Show this help message"
    echo
    echo "Directory structure:"
    echo "  <experiments-dir>/"
    echo "  └── <experiment-name>/"
    echo "      └── runs/"
    echo "          └── <run-name>/"
    echo "              ├── input/          # Input data"
    echo "              ├── output/         # Output data and metrics"
    echo "              ├── model/          # Saved model weights"
    echo "              └── checkpoints/    # Model checkpoints"
}

# Add help flag handling
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            break
            ;;
    esac
done


# Default values
EXPERIMENTS_DIR="/home/jeff/data/fpe/experiments"
EXPERIMENT_NAME="test-experiment"
RUN_NAME="test-run"
IMAGES_DIR="/home/jeff/data/fpe/images"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiments-dir)
            EXPERIMENTS_DIR="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --images-dir)
            IMAGES_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

RUN_DIR="${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/runs/${RUN_NAME}"

# Validate directories
if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Run directory not found: $RUN_DIR"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found: $IMAGES_DIR"
    exit 1
fi

DATA_DIR="${RUN_DIR}/input"
OUTPUT_TRAIN_DIR="${RUN_DIR}/output/train"
CHECKPOINTS_DIR="${RUN_DIR}/checkpoints"
MODEL_DIR="${RUN_DIR}/model"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_TRAIN_DIR"
mkdir -p "$CHECKPOINTS_DIR"
mkdir -p "$MODEL_DIR"

echo "Starting experiment run: $RUN_NAME"

# Run training
echo "Starting training..."
docker run -it \
    --runtime=nvidia \
    --gpus all \
    -v "${DATA_DIR}:/opt/ml/input/data/data" \
    -v "${IMAGES_DIR}:/opt/ml/input/data/images" \
    -v "${OUTPUT_TRAIN_DIR}:/opt/ml/output/data" \
    -v "${CHECKPOINTS_DIR}:/opt/ml/checkpoints" \
    -v "${MODEL_DIR}:/opt/ml/model" \
    -v "$(pwd):/app" \
    -w /app \
    --env-file .env.docker \
    --network host \
    --shm-size=4g \
    fpe-rank \
    python src/train_aux.py \
        --mlflow-experiment-name "${EXPERIMENT_NAME}" \
        --mlflow-run-name "${RUN_NAME}" \
        --config /opt/ml/input/data/data/config.yaml

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

# Run testing
OUTPUT_TEST_DIR="${RUN_DIR}/output/test"

# Create directories if they don't exist
mkdir -p "$OUTPUT_TEST_DIR"

echo "Starting testing..."
docker run -it \
    --runtime=nvidia \
    --gpus all \
    -v "${DATA_DIR}:/opt/ml/input/data/data" \
    -v "${IMAGES_DIR}:/opt/ml/input/data/images" \
    -v "${OUTPUT_TEST_DIR}:/opt/ml/output/data" \
    -v "${MODEL_DIR}:/opt/ml/model" \
    -v "$(pwd):/app" \
    -w /app \
    --env-file .env.docker \
    --network host \
    --shm-size=4g \
    fpe-rank \
    python src/test_aux.py \
        --mlflow-experiment-name "${EXPERIMENT_NAME}" \
        --run-name "${RUN_NAME}"

# Check if testing was successful
if [ $? -ne 0 ]; then
    echo "Error: Testing failed"
    exit 1
fi

echo "Run completed successfully"
echo "MLflow run name: $RUN_NAME"
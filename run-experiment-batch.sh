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
    echo "  --run-file FILE          File listing all runs (default: runs.txt)"
    echo "  --images-dir DIR         Directory containing image files (default: /home/jeff/data/fpe/images)"
    echo "  -h, --help               Show this help message"
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
RUNS_FILE="runs.txt"
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
        --runs-file)
            RUNS_FILE="$2"
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

RUNS_PATH="${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/${RUNS_FILE}"

# read in list of run names from runs.txt
RUN_NAMES=$(cat $RUNS_PATH)

for RUN_NAME in $RUN_NAMES; do
    echo "Running $RUN_NAME"
    ./run-experiment.sh --experiments-dir $EXPERIMENTS_DIR --experiment-name $EXPERIMENT_NAME --run-name $RUN_NAME --images-dir $IMAGES_DIR
done

#!/bin/bash

RUN="06"
STATIONS="12 16 46 166"
TRIALS="1 2 3 4 5"
# N_PAIRS="50 100 500"
N_PAIRS="200"

for station in $STATIONS; do
    for trial in $TRIALS; do
        for n_pairs in $N_PAIRS; do
            RUN_NAME="n${n_pairs}_t${trial}_s${station}"
            echo "Running $RUN_NAME"

            IMAGES_DIR="/home/jeff/data/fpe/images"
            DATA_DIR="/home/jeff/git/fpe-model/experiments/20250107-multisite/runs/run-06/${RUN_NAME}/input"
            OUTPUT_DIR="/home/jeff/git/fpe-model/experiments/20250107-multisite/runs/run-06/${RUN_NAME}/output"
            MODEL_DIR="${OUTPUT_DIR}/model"

            PREDICTION_FILE="${OUTPUT_DIR}/data/predictions.csv"

            if [ -f "$PREDICTION_FILE" ]; then
                echo "Prediction file already exists: $PREDICTION_FILE"
                continue
            fi

            # echo "DATA_DIR: $DATA_DIR"
            # echo "IMAGES_DIR: $IMAGES_DIR"
            # echo "MODEL_DIR: $MODEL_DIR"
            # echo "OUTPUT_DIR: $OUTPUT_DIR"

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
            CMD_ARGS="python train-rank.py --pretrained-model-path /opt/ml/input/data/data/model.pth --load-optimizer-state --early-stopping-min-delta 0.001 --mlflow-experiment-name 20250107-multisite --mlflow-run-name run06-${RUN_NAME}"

            echo $CMD_ARGS
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
        done
    done
done

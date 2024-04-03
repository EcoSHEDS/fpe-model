#!/bin/bash

# batch run training jobs
# usage: ./batch-run-train.sh <stations file> <path/to/datasets> <model code>
# example: ./batch-run-train.sh /mnt/d/fpe/rank/stations.txt /mnt/d/fpe/rank RANK-FLOW-20240402

set -eu

STATIONS_FILE="$1"
DIRECTORY="$2"
MODEL_CODE="$6"

while IFS= read -r LINE
do
    STATION_ID=$(echo "${LINE}" | awk '{print $1}')
    ARGS="--station-id ${STATION_ID} --directory=${DIRECTORY} --model-code ${MODEL_CODE}"
    echo "running: python src/run-train.py ${ARGS}"
    #python src/run-train.py ${ARGS}
done < ${STATIONS_FILE}

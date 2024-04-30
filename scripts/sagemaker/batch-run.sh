#!/bin/bash

# batch run model jobs
# usage: ./batch-run.sh <method> <stations file> <path/to/datasets> <model code>
# example: ./batch-run.sh train /mnt/d/fpe/rank/stations.txt /mnt/d/fpe/rank RANK-FLOW-20240410

set -eu

METHOD="$1"
STATIONS_FILE="$2"
DIRECTORY="$3"
MODEL_CODE="$4"

while IFS= read -r LINE
do
    STATION_ID=$(echo "${LINE}" | awk '{print $1}')
    ARGS="--station-id ${STATION_ID} --directory=${DIRECTORY} --model-code ${MODEL_CODE}"
    SCRIPT="src/run-${METHOD}.py"
    echo "running: python ${SCRIPT} ${ARGS}"
    python ${SCRIPT} ${ARGS}
done < ${STATIONS_FILE}

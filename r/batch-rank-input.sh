#!/bin/bash
# batch export datasets for multiple stations
# usage: ./batch-rank-input.sh <stations file> <variable_id> <path/to/datasets> <dataset version> <max annotation date>
# example: ./batch-rank-input.sh /d/fpe/datasets/stations.txt FLOW_CFS /d/fpe/datasets 20240327 2023-08-31

set -eu

STATIONS_FILE="$1"
VARIABLE_ID="$2"
OUT_DIR="$3"
DATASET_VERSION="$4"
MAX_DATE="$5"

while IFS= read -r LINE
do
    STATION_ID=$(echo "${LINE}" | awk '{print $1}')
    ARGS="-s ${STATION_ID} -v ${VARIABLE_ID} -d ${OUT_DIR} -o --dataset-version=${DATASET_VERSION} --annotations-end=${MAX_DATE}"
    echo "running: Rscript rank-input.R ${ARGS}"
    Rscript rank-input.R ${ARGS}
done < ${STATIONS_FILE}

#!/bin/bash
# batch export datasets for multiple stations
# usage: ./batch-rank-input.sh <stations file> <path/to/datasets> <variable_id> <dataset version> <model code> # <max annotation date>
# example: ./batch-rank-input.sh /mnt/d/fpe/rank/stations.txt /mnt/d/fpe/rank FLOW_CFS RANK-FLOW-20240402 RANK-FLOW-20240402 # 2023-09-30

set -eu

STATIONS_FILE="$1"
OUT_DIR="$2"
VARIABLE_ID="$3"
DATASET_CODE="$4"
MODEL_CODE="$5"
# MAX_DATE="$6"

while IFS= read -r LINE
do
    STATION_ID=$(echo "${LINE}" | awk '{print $1}')
    # ARGS="-s ${STATION_ID} -v ${VARIABLE_ID} -d ${OUT_DIR} -o -D ${DATASET_CODE} --annotations-end=${MAX_DATE} ${MODEL_CODE}"
    ARGS="-s ${STATION_ID} -v ${VARIABLE_ID} -d ${OUT_DIR} -o -D ${DATASET_CODE} ${MODEL_CODE}"
    echo "running: Rscript rank-input.R ${ARGS}"
    Rscript rank-input.R ${ARGS}
done < ${STATIONS_FILE}

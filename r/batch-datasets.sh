#!/bin/bash
# batch export datasets for multiple stations
# usage: ./batch-datasets.sh <stations file> <variable_id> <path/to/datasets>
# example: ./batch-datasets.sh /d/fpe/datasets/stations.txt FLOW_CFS /d/fpe/datasets

STATIONS_FILE="$1"
VARIABLE_ID="$2"
OUT_DIR="$3"

while IFS= read -r LINE
do
    STATION_ID=$(echo "${LINE}" | awk '{print $1}')
    ARGS="${STATION_ID} ${VARIABLE_ID} ${OUT_DIR}"
    echo "running: ${ARGS}"
    Rscript dataset.R ${ARGS}
done < ${STATIONS_FILE}

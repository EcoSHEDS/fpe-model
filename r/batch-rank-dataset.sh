#!/bin/bash
# batch export datasets for multiple stations
# usage: ./batch-rank-dataset.sh <stations file> <variable_id> <path/to/datasets>
# example: ./batch-rank-dataset.sh /d/fpe/datasets/stations.txt FLOW_CFS /d/fpe/datasets

set -eu

STATIONS_FILE="$1"
VARIABLE_ID="$2"
OUT_DIR="$3"

while IFS= read -r LINE
do
    STATION_ID=$(echo "${LINE}" | awk '{print $1}')
    ARGS="-s ${STATION_ID} -v ${VARIABLE_ID} -d ${OUT_DIR} -o"
    echo "running: Rscript rank-dataset.R ${ARGS}"
    Rscript rank-dataset.R ${ARGS}
done < ${STATIONS_FILE}

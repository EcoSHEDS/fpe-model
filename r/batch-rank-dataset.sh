#!/bin/bash
# batch export datasets for multiple stations
# usage: ./batch-rank-dataset.sh <stations file> <datasets dir> <variable_id> <dataset code>
# example: ./batch-rank-dataset.sh /mnt/d/fpe/rank/stations.txt /mnt/d/fpe/rank FLOW_CFS RANK-FLOW-20240402

set -eu

STATIONS_FILE="$1"
OUT_DIR="$2"
VARIABLE_ID="$3"
DATASET_CODE="$4"

while IFS= read -r LINE
do
    STATION_ID=$(echo "${LINE}" | awk '{print $1}')
    ARGS="-s ${STATION_ID} -v ${VARIABLE_ID} -d ${OUT_DIR} -o ${DATASET_CODE}"
    echo "running: Rscript rank-dataset.R ${ARGS}"
    Rscript rank-dataset.R ${ARGS}
done < ${STATIONS_FILE}

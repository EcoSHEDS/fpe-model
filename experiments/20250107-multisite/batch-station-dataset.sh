#!/bin/bash
# batch export datasets for multiple stations
# usage: ./batch-station-dataset.sh <stations file> <stations dir> <variable_id>
# example: ./batch-station-dataset.sh /mnt/d/fpe/path/to/stations.txt /mnt/d/fpe/stations FLOW_CFS

set -eu

STATIONS_FILE="$1"
OUT_DIR="$2"
VARIABLE_ID="$3"

while IFS= read -r LINE
do
    STATION_ID=$(echo "${LINE}" | awk '{print $1}')
    ARGS="-s ${STATION_ID} -v ${VARIABLE_ID} -d ${OUT_DIR} -o"
    echo "running: Rscript station-dataset.R ${ARGS}"
    Rscript station-dataset.R ${ARGS}
done < ${STATIONS_FILE}

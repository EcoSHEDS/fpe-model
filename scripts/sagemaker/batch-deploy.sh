#!/bin/bash

# batch deploy models to s3
# usage: ./batch-deploy.sh <stations+uuid file> <path/to/datasets> <model code>
# example: ./batch-deploy.sh /mnt/d/fpe/rank/stations-model-uuid.txt /mnt/d/fpe/rank RANK-FLOW-20240410

set -eu

STATIONS_FILE="$1"
DIRECTORY="$2"
MODEL_CODE="$3"

while IFS= read -r LINE
do
    STATION_ID=$(echo "${LINE}" | awk '{print $1}')
    UUID=$(echo "${LINE}" | awk '{print $2}')
    echo "${STATION_ID} | ${UUID}"
    DIAGNOSTICS_S3="s3://usgs-chs-conte-prod-fpe-storage/models/${UUID}/${MODEL_CODE}.html"
    DIAGNOSTICS_FILE="${DIRECTORY}/${STATION_ID}/models/${MODEL_CODE}/${MODEL_CODE}.html"
    PREDICTIONS_S3="s3://usgs-chs-conte-prod-fpe-storage/models/${UUID}/predictions.csv"
    PREDICTIONS_FILE="${DIRECTORY}/${STATION_ID}/models/${MODEL_CODE}/transform/predictions.csv"
    echo "uploading: ${DIAGNOSTICS_FILE} to ${DIAGNOSTICS_S3}"
    aws s3 cp ${DIAGNOSTICS_FILE} ${DIAGNOSTICS_S3}
    echo "uploading: ${PREDICTIONS_FILE} to ${PREDICTIONS_S3}"
    aws s3 cp ${PREDICTIONS_FILE} ${PREDICTIONS_S3}
done < ${STATIONS_FILE}

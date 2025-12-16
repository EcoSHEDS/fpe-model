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

    ANNOTATIONS_S3="s3://usgs-chs-conte-prod-fpe-storage/models/${UUID}/annotations.csv"
    ANNOTATIONS_FILE="${DIRECTORY}/${STATION_ID}/models/${MODEL_CODE}/input/annotations.csv"
    aws s3 cp ${ANNOTATIONS_FILE} ${ANNOTATIONS_S3}

    IMAGES_S3="s3://usgs-chs-conte-prod-fpe-storage/models/${UUID}/images.csv"
    IMAGES_FILE="${DIRECTORY}/${STATION_ID}/models/${MODEL_CODE}/input/images.csv"
    aws s3 cp ${IMAGES_FILE} ${IMAGES_S3}

    MODEL_S3="s3://usgs-chs-conte-prod-fpe-storage/models/${UUID}/model.tar.gz"
    MODEL_FILE="${DIRECTORY}/${STATION_ID}/models/${MODEL_CODE}/output/model.tar.gz"
    aws s3 cp ${MODEL_FILE} ${MODEL_S3}

    DIAGNOSTICS_S3="s3://usgs-chs-conte-prod-fpe-storage/models/${UUID}/${MODEL_CODE}.html"
    DIAGNOSTICS_FILE="${DIRECTORY}/${STATION_ID}/models/${MODEL_CODE}/${MODEL_CODE}.html"
    aws s3 cp ${DIAGNOSTICS_FILE} ${DIAGNOSTICS_S3}

    PREDICTIONS_S3="s3://usgs-chs-conte-prod-fpe-storage/models/${UUID}/predictions.csv"
    PREDICTIONS_FILE="${DIRECTORY}/${STATION_ID}/models/${MODEL_CODE}/transform/predictions.csv"
    aws s3 cp ${PREDICTIONS_FILE} ${PREDICTIONS_S3}
done < ${STATIONS_FILE}

#!/bin/bash

MODELS="concat encoder lstm-d-90 lstm-h-30"
STATIONS="12 29"

for model in $MODELS; do
    for station in $STATIONS; do
        echo "Running model $model for station $station"
        ./run-docker.sh --model $model --station-id $station
    done
done

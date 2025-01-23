#!/bin/bash

MODELS="none concat-d concat-h encoder-d encoder-h lstm-d lstm-h"
STATIONS="12 13 14 15 29 65 68 80 81 89 90 95"

for model in $MODELS; do
    for station in $STATIONS; do
        echo "Running model $model for station $station"
        ./run-docker.sh --model $model --station-id $station
    done
done

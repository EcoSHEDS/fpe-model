#!/bin/bash

MODELS="encoder lstm-d-90 lstm-h-30"
TYPES="ranknet regression"
N_PAIRS="500 1000 5000"
STATIONS="12 13 14 15 29 65 68 80 81 89 90 95"

for model in $MODELS; do
    for n_pairs in $N_PAIRS; do
        for type in $TYPES; do
            for station in $STATIONS; do
                echo "Running model $model of type $type for station $station with $n_pairs pairs"
                ./run-docker.sh --model $model --station-id $station --n-pairs $n_pairs --type $type
            done
        done
    done
done

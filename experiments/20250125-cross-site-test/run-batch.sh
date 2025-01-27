#!/bin/bash

STATIONS="12 13 14 15 29 65 68 80 81 89 90 95"

for station in $STATIONS; do
    echo "Running for station $station"
    ./run-docker.sh --station-id $station
done

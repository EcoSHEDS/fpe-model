#!/bin/bash

# ./run-experiment.sh --experiments-dir ~/data/fpe/experiments --experiment-name 20241127-WB0-AVERYBB --run-name train_both_mix --images-dir ~/data/fpe/images
# ./run-experiment.sh --experiments-dir ~/data/fpe/experiments --experiment-name 20241127-WB0-AVERYBB --run-name train_AVERYBB_fine_WB0 --images-dir ~/data/fpe/images
# ./run-experiment.sh --experiments-dir ~/data/fpe/experiments --experiment-name 20241127-WB0-AVERYBB --run-name train_WB0_fine_AVERYBB --images-dir ~/data/fpe/images

experiments="20241127-AVERYBB-arch 20241127-AVERYBB-transform"

for experiment in $experiments; do
    echo "Running experiment: $experiment"
    ./run-experiment-batch.sh --experiments-dir ~/data/fpe/experiments --experiment-name $experiment --runs-file runs.txt --images-dir ~/data/fpe/images
done

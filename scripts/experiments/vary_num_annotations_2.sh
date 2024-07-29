#!/bin/bash

# Paths
DATA_ROOT="/mnt/blobfuse/usgs-streamflow/fpe_stations"
ANNOTATIONS_ROOT="$HOME/azurefiles/projects/streamflow/jeff_data/stations"
REPO_ROOT="$HOME/ssdprivate/repos/fpe-model"
SPLIT_SEED=1632

# station_data_folder="Avery Brook_River Left_01171000"
# station_annots_folder="15-Avery Brook_River Left_01171000"

# station_data_folder="Avery Brook_River Right_01171000"
# station_annots_folder="14-Avery Brook_River Right_01171000"

# station_data_folder="Avery Brook_Side_01171000"
# station_annots_folder="13-Avery Brook_Side_01171000"

station_data_folder="Sanderson Brook_01171010"
station_annots_folder="9-Sanderson Brook_01171010"

# Check if the expected folders exist
if [ ! -d "$DATA_ROOT/$station_data_folder" ]; then
    echo "Data folder does not exist: $DATA_ROOT/$station_data_folder"
    exit 1
fi

if [ ! -d "$ANNOTATIONS_ROOT/$station_annots_folder" ]; then
    echo "Annotations folder does not exist: $ANNOTATIONS_ROOT/$station_annots_folder"
    exit 1
fi

# Values to iterate over
# values=(100 200 300 400 500 750 1000 1250 1500 1817)
# values=(100 200 300 400 500 750 1000 1250 1500 1773)
# values=(100 200 300 400 500 750 1000 1250 1500 1955)
values=(100 200 300 400 500 750 1000 1250 1500 2000 2500 3000 3856)

for value in "${values[@]}"; do
    # Construct the command using an array with the current value
    command=(
        python "$REPO_ROOT/scripts/train.py"
        --images-dir "$DATA_ROOT/$station_data_folder/FLOW_CFS/"
        --pairs-file "$ANNOTATIONS_ROOT/$station_annots_folder/input_$SPLIT_SEED/pairs-train_$value.csv"
        --gpu 3
        --output-dir "$REPO_ROOT/results/vary_num_annotations/jeff_splits-split_seed_$SPLIT_SEED/$station_annots_folder/pairs-train_$value"
        --model-dir "$REPO_ROOT/results/vary_num_annotations/jeff_splits-split_seed_$SPLIT_SEED/$station_annots_folder/pairs-train_$value/model"
        --augment --normalize
    )

    # Print the command for debugging
    printf "%q " "${command[@]}"
    echo

    # Run the command
    "${command[@]}"
done

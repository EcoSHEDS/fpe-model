#!/bin/bash

# Paths
DATA_ROOT="$HOME/ssdprivate/data/usgs-streamflow/fpe_stations"
ANNOTATIONS_ROOT="$HOME/azurefiles/projects/streamflow/jeff_data/stations"
REPO_ROOT="$HOME/ssdprivate/repos/fpe-model"
SPLIT_SEED=3274 #4444 #8436 #1632 #2927 #


# Define associative arrays for station data and annotation folders
declare -A station_data_folders=(
    ["Avery Brook Bridge"]="Avery Brook_Bridge_01171000"
    ["Avery Brook River Left"]="Avery Brook_River Left_01171000"
    ["Avery Brook River Right"]="Avery Brook_River Right_01171000"
    ["Avery Brook Side"]="Avery Brook_Side_01171000"
    ["Green River"]="Green River_01170100"
    ["Sanderson Brook"]="Sanderson Brook_01171010"
    ["West Branch Swift River"]="West Branch Swift River_01174565"
    ["West Brook Lower"]="West Brook Lower_01171090"
    ["West Brook Reservoir"]="West Brook Reservoir_01171020"
    ["West Brook 0"]="West Brook 0_01171100"
    ["West Whately"]="West Whately_01171005"
)

declare -A station_annots_folders=(
    ["Avery Brook Bridge"]="12-Avery Brook_Bridge_01171000"
    ["Avery Brook River Left"]="15-Avery Brook_River Left_01171000"
    ["Avery Brook River Right"]="14-Avery Brook_River Right_01171000"
    ["Avery Brook Side"]="13-Avery Brook_Side_01171000"
    ["Green River"]="65-Green River_01170100"
    ["Sanderson Brook"]="9-Sanderson Brook_01171010"
    ["West Branch Swift River"]="68-West Branch Swift River_01174565"
    ["West Brook Lower"]="10-West Brook Lower_01171090"
    ["West Brook Reservoir"]="16-West Brook Reservoir_01171020"
    ["West Brook 0"]="29-West Brook 0_01171100"
    ["West Whately"]="17-West Whately_01171005"
)

# Define the values to iterate over for each site
declare -A values=(
    ["Avery Brook Bridge"]="100 200 300 400 500 750 1000 1250 1500 2000 2500 2512"
    ["Avery Brook River Left"]="100 200 300 400 500 750 1000 1250 1500 1817"
    ["Avery Brook River Right"]="100 200 300 400 500 750 1000 1250 1500 1773"
    ["Avery Brook Side"]="100 200 300 400 500 750 1000 1250 1500 1955"
    ["Green River"]="100 200 300 400 500 750 1000 1250 1500 2000 2500 3000 4000 4059"
    ["Sanderson Brook"]="100 200 300 400 500 750 1000 1250 1500 2000 2500 3000 3856"
    ["West Branch Swift River"]="100 200 300 400 500 750 1000 1250 1500 2000 2500 2838"
    ["West Brook Lower"]="100 200 300 400 500 750 1000 1250 1500 1809"
    ["West Brook Reservoir"]="100 200 300 400 500 750 1000 1250 1500 1862"
    ["West Brook 0"]="100 200 300 400 500 750 1000 1250 1500 2000 2500 3000 4000 6365"
    ["West Whately"]="100 200 300 400 500 750 1000 1250 1500 2000 2007"
)

# Iterate over each site
for site in "${!station_data_folders[@]}"; do
    station_data_folder="${station_data_folders[$site]}"
    station_annots_folder="${station_annots_folders[$site]}"
    site_values=(${values[$site]})

    # Check if the expected folders exist
    if [ ! -d "$DATA_ROOT/$station_data_folder" ]; then
        echo "Data folder does not exist: $DATA_ROOT/$station_data_folder"
        exit 1
    fi

    if [ ! -d "$ANNOTATIONS_ROOT/$station_annots_folder" ]; then
        echo "Annotations folder does not exist: $ANNOTATIONS_ROOT/$station_annots_folder"
        exit 1
    fi

    # Iterate over the values for the current site
    for value in "${site_values[@]}"; do
        # Construct the command using an array with the current value
        command=(
            python "$REPO_ROOT/scripts/predict.py"
            --images-dir "$DATA_ROOT/$station_data_folder/FLOW_CFS/"
            --pairs-file "$ANNOTATIONS_ROOT/$station_annots_folder/input_$SPLIT_SEED/pairs-train_$value.csv"
            --output-dir "$REPO_ROOT/results/vary_num_annotations/jeff_splits-split_seed_$SPLIT_SEED/$station_annots_folder/pairs-train_$value"
            --model-dir "$HOME/azurefiles/projects/streamflow/results/vary_num_annotations/no_early_stopping_max_50_epochs/jeff_splits-split_seed_$SPLIT_SEED/$station_annots_folder/pairs-train_$value/model"
            --augment --normalize --epochs 1
            --gpu 3
        )

        # Print the command for debugging
        printf "%q " "${command[@]}"
        echo

        # Run the command
        "${command[@]}"

        # Pause for 5 seconds
        sleep 5
    done
done
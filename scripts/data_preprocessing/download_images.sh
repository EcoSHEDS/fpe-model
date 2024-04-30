#!/bin/bash

# Load environment variables from .env file
current_dir=$(pwd)
source "${current_dir}/../../conf/local/.env"

# Define the arguments to pass to the script
dataset_dirs=("Avery Brook_Bridge_01171000/FLOW_CFS")
# dataset_dirs=("Avery Brook_River Left_01171000/FLOW_CFS")
# dataset_dirs=("Avery Brook_River Right_01171000/FLOW_CFS")
# dataset_dirs=("Avery Brook_Side_01171000/FLOW_CFS")
# dataset_dirs=("Sanderson Brook_01171010/FLOW_CFS")
# dataset_dirs=("West Branch Swift River_01174565/FLOW_CFS")
# dataset_dirs=("West Brook 0_01171100/FLOW_CFS")
# dataset_dirs=("West Brook Lower_01171090/FLOW_CFS")
# dataset_dirs+=("West Brook Reservoir_01171020/FLOW_CFS")
# dataset_dirs=("West Brook Upper_01171030/FLOW_CFS")
# dataset_dirs=("West Whately_01171005/FLOW_CFS")

# Run the script with each set of arguments
for i in "${!dataset_dirs[@]}"; do
    cmd="python download_images.py --dataset-dir \"${FPE_DATA_ROOT}/${dataset_dirs[$i]}\" --csv-file \"images.csv\" --output-dir \"\" --num-workers 16 --max-attempts 3"
    echo $cmd
    eval $cmd
done
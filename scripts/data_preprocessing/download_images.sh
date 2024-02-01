#!/bin/bash

# Load environment variables from .env file
current_dir=$(pwd)
source "${current_dir}/../../conf/local/.env"

# Define the arguments to pass to the script
dataset_dirs=("West Whately_01171005/FLOW_CFS")

# Run the script with each set of arguments
for i in "${!dataset_dirs[@]}"; do
    cmd="python download_images.py --dataset-dir \"${FPE_DATA_ROOT}/${dataset_dirs[$i]}\" --csv-file \"images.csv\" --output-dir \"\" --num-workers 8 --max-attempts 3"
    echo $cmd
    eval $cmd
done
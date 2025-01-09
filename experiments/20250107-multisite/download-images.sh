#!/bin/bash

# Check if required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input-file> <s3-bucket-name>"
    echo "Example: $0 image-paths.txt my-bucket-name"
    exit 1
fi

INPUT_FILE="$1"
BUCKET_NAME="$2"
MAX_PARALLEL=10  # Maximum number of parallel downloads

# Function to download a single file
download_file() {
    local path="$1"
    local bucket="$2"

    if [ -f "$path" ]; then
        # echo skipping
        return 0
    fi

    # Create the directory structure
    local dir=$(dirname "$path")
    mkdir -p "$dir"

    # Download the file
    aws s3 cp "s3://$bucket/$path" "$path"
}
export -f download_file

# Read the file and process each line in parallel
cat "$INPUT_FILE" | xargs -I {} -P "$MAX_PARALLEL" bash -c 'download_file "$@"' _ {} "$BUCKET_NAME"

echo "Download completed successfully!"

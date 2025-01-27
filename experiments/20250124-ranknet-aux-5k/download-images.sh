#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input-file> <s3-bucket-name> [--images-root <dir>]"
    echo "Example: $0 image-paths.txt my-bucket-name --images-root /path/to/images"
    exit 1
fi

INPUT_FILE="$1"
BUCKET_NAME="$2"
IMAGES_ROOT="."  # Default to current directory
MAX_PARALLEL=20  # Maximum number of parallel downloads

# Parse optional arguments
shift 2
while [[ $# -gt 0 ]]; do
    case $1 in
        --images-root)
            IMAGES_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 <input-file> <s3-bucket-name> [--images-root <dir>]"
            exit 1
            ;;
    esac
done

# Function to download a single file
download_file() {
    local path="$1"
    local bucket="$2"
    local root="$3"

    local full_path="${root}/${path}"
    if [ -f "$full_path" ]; then
        # echo skipping
        return 0
    fi

    # Create the directory structure
    local dir=$(dirname "$full_path")
    mkdir -p "$dir"

    # Download the file
    aws s3 cp "s3://$bucket/$path" "$full_path"
}
export -f download_file

# Read the file and process each line in parallel
cat "$INPUT_FILE" | xargs -I {} -P "$MAX_PARALLEL" bash -c 'download_file "$@"' _ {} "$BUCKET_NAME" "$IMAGES_ROOT"

echo "Download completed successfully!"

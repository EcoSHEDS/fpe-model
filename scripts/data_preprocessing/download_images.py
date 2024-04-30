""" Download images from URLs in a CSV file.

This script downloads images from URLs in a CSV file and saves them to a
specified directory. It also checks that the downloaded images are valid.

The CSV file should have the following columns:
    - URL: URL to download image from
    - filename: filename to save image as

Example usage:
    python download_images.py \
        --dataset-dir /path/to/fpe_stations/AVERYBB \
        --csv-file AVERYBB-20230829/data/flow-images.csv \
        --output-dir '' \
        --num-workers 8 \
        --max-attempts 3

This opens /path/to/fpe_stations/AVERYBB/AVERYBB-20230829/data/flow-images.csv
and downloads images to /path/to/fpe_stations/AVERYBB/ with the same filename
as in the CSV file. In this case, the CSV file has filenames starting with
imagesets/ so images are saved to /path/to/fpe_stations/AVERYBB/imagesets/...
This allows multiple snapshots of the same station to be saved without
duplicating image files.

"""

import argparse
import concurrent.futures  # for multithreading
from pathlib import Path
from typing import Union

import pandas as pd
from tqdm import tqdm

from src.utils import check_image, download_image_from_url


def download_and_check_image(
    url: str, path: Union[str, Path], attempts: int = 3
) -> bool:
    # Check if file exists and is valid
    if path.exists() and check_image(str(path)):
        return True

    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Try to download and check the image
    for i in range(attempts):
        if download_image_from_url(url, str(path)) and check_image(str(path)):
            return True
        print(f"Attempt {i + 1} failed for {path}. Retrying...")

    # If function hasn't returned, download or check has failed
    print(f"Failed to download or validate {path} after {attempts} attempts.")
    return False


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Downloads and validates images from a CSV file."
    )
    parser.add_argument(
        "--dataset-dir", required=True, help="Directory where the dataset is located."
    )
    parser.add_argument(
        "--csv-file",
        required=True,
        help="Relative path to the CSV file with image URLs and filenames.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Relative path to the directory to save images.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads for downloading images.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max attempts to download an image before giving up.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datafile = Path(args.dataset_dir) / args.csv_file
    outdir = Path(args.dataset_dir) / args.output_dir
    with open(datafile, "r") as f:
        df = pd.read_csv(f)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        futures = [
            executor.submit(
                download_and_check_image,
                row["url"],
                outdir / row["filename"],
                args.max_attempts,
            )
            for _, row in df.iterrows()
        ]

        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    print("Done.")

""" Download images from urls in a csv file.

This script downloads images from urls in a csv file and saves them to a
specified directory. It also checks that the downloaded images are valid.

The csv file should have the following columns:
    - url: url to download image from
    - filename: filename to save image as

Example usage:
    python download_images.py \
        --data-root-dir /path/to/fpe_stations/AVERYBB \
        --data-filename AVERYBB-20230829/data/flow-images.csv \
        --image-base-dir ''

This opens /path/to/fpe_stations/AVERYBB/AVERYBB-20230829/data/flow-images.csv
and downloads images to /path/to/fpe_stations/AVERYBB/ with the same filename
as in the csv file. In this case, the csv file has filenames starting with
imagesets/ so images are saved to/path/to/fpe_stations/AVERYBB/imagesets/...
This allows multiple snapshots of the same station to be saved without
duplicating image files.

"""

import argparse
import os
import shutil  # save img locally

import pandas as pd
import requests  # request img from web
from torchvision.io import read_image
from tqdm import tqdm


def download_image_from_url(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
    else:
        raise Exception(f"Image couldn't be retrieved from URL: \n\t{url}")


def check_downloaded_image(image_path, url):
    try:
        read_image(image_path)
    except Exception as e:
        print(f"Cannot read image {image_path}")
        print(e)
        print("Retrying download...")
        download_image_from_url(url, image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data-root-dir", required=True, help="Dataset root directory")
    parser.add_argument(
        "--data-filename",
        required=True,
        help="CSV file containing image urls",
    )
    parser.add_argument(
        "--image-base-dir",
        required=True,
        help="Directory within dataset root directory to save images",
    )
    args = parser.parse_args()

    datafile = os.path.join(args.data_root_dir, args.data_filename)
    df = pd.read_csv(datafile)
    outdir = os.path.join(args.data_root_dir, args.image_base_dir)

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        url = row["url"]
        out_file_name = row["filename"]
        out_file_path = os.path.join(outdir, out_file_name)

        if not os.path.exists(out_file_path):
            os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
            download_image_from_url(url, out_file_path)

    print("Checking downloaded images...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        url = row["url"]
        out_file_name = row["filename"]
        out_file_path = os.path.join(outdir, out_file_name)
        check_downloaded_image(out_file_path, url)
    print("Done.")

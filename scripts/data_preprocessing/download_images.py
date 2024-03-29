import os
import argparse
import requests  # request img from web
import shutil  # save img locally
import pandas as pd
from tqdm import tqdm
from torchvision.io import read_image

# from PIL import Image


def download_image_from_url(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
    else:
        raise Exception(f"Image couldn't be retrieved from URL: \n\t{url}")


def check_downloaded_image(image_path, url):
    try:
        image = read_image(image_path)
    except Exception as e:
        print(f"Cannot read image {image_path}")
        print(e)
        print("Retrying download...")
        download_image_from_url(url, image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data-root-dir", required=True, help="random seed")
    parser.add_argument("--data-filename", required=True, help="random seed")
    parser.add_argument("--image-base-dir", required=True, help="random seed")
    args = parser.parse_args()

    datafile = os.path.join(args.data_root_dir, args.data_filename)
    df = pd.read_csv(datafile)
    outdir = os.path.join(args.data_root_dir, args.image_base_dir)

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        url = row["url"]
        out_file_name = row["filename"]
        out_file_path = os.path.join(outdir, out_file_name)

        if not os.path.exists(out_file_path):
            download_image_from_url(url, out_file_path)

    print("Checking downloaded images...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        url = row["url"]
        out_file_name = row["filename"]
        out_file_path = os.path.join(outdir, out_file_name)
        check_downloaded_image(out_file_path, url)
    print("Done.")

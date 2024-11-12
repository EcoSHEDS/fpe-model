import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.transform import load_images_from_csv, model_fn, predict_fn
from src.datasets import FlowPhotoDataset

def transform(args):
    data_filepath = os.path.join(args.values_dir, args.data_file)
    df = load_images_from_csv(data_filepath)
    df["score"] = np.nan
    ds = FlowPhotoDataset(df, args.images_dir)

    model = model_fn(args.model_dir)
    model.eval()

    print(f"computing predictions (n={len(ds)})")
    with torch.no_grad():
        for idx, image in tqdm(enumerate(ds), total=len(ds)):
            pred = predict_fn(image[0], model)
            df.at[idx, "score"] = pred["score"]

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ["FPE_MODEL_DIR"])
    parser.add_argument("--output-dir", type=str, default=os.environ["FPE_OUTPUT_DIR"])
    parser.add_argument("--images-dir", type=str, default=os.environ["FPE_IMAGES_DIR"])
    parser.add_argument("--values-dir", type=str, default=os.environ["FPE_VALUES_DIR"])
    parser.add_argument(
        "--data-file",
        type=str,
        default="images.csv",
        help="filename of images CSV file",
    )

    args = parser.parse_args()
    results = transform(args)

    output_file = os.path.join(args.output_dir, "data", "predictions.csv")
    print(f"saving predictions: {output_file}")
    results.to_csv(os.path.join(output_file))

import argparse
import os
import json
import pandas as pd
from utils import (
    set_seeds,
    load_data,
)
from datasets import (
    RandomStratifiedWindowFlow,
    FlowPhotoRankingDataset,
    random_pairs,
)

def generate_datasets(args):
    print("args:")
    print(args.__dict__)

    print("set seed")
    set_seeds(args.random_seed)

    print("loading data file")
    df = load_data(args.data_file)

    output_dir = args.output_dir
    if (output_dir is None):
        output_dir = os.path.join(os.path.dirname(args.data_file), "splits")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("split flow-images into train/val/test using random stratified week")
    try:
        splits = RandomStratifiedWindowFlow().split(df, 0.8, 0.1, 0.1)
    except ValueError:
        print("ValueError: not enough data, trying again with window='day'")
        splits = RandomStratifiedWindowFlow().split(df, 0.8, 0.1, 0.1, window="day")

    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    print("saving flow-images splits to csv")
    train_df.to_csv(os.path.join(output_dir, "images-train.csv"))
    val_df.to_csv(os.path.join(output_dir, "images-val.csv"))
    test_df.to_csv(os.path.join(output_dir, "images-test.csv"))

    print("creating flow photo ranking datasets")
    train_ds = FlowPhotoRankingDataset(train_df, os.path.dirname(args.data_file))
    val_ds = FlowPhotoRankingDataset(val_df, os.path.dirname(args.data_file))
    test_ds = FlowPhotoRankingDataset(test_df, os.path.dirname(args.data_file))

    print("ranking image pairs by oracle")
    train_ds.rank_image_pairs(
        random_pairs,
        args.num_train_pairs,
        args.margin,
        args.margin_mode,
    )
    val_ds.rank_image_pairs(
        random_pairs,
        args.num_eval_pairs,
        args.margin,
        args.margin_mode,
    )
    test_ds.rank_image_pairs(
        random_pairs,
        args.num_eval_pairs,
        args.margin,
        args.margin_mode,
    )

    print("saving image pairs to csv")
    train_pairs = [
        {
            "image_id_1": train_ds.table.iloc[idx1][train_ds.cols['image_id']],
            "timestamp_1": train_ds.table.iloc[idx1][train_ds.cols['timestamp']],
            "filename_1": train_ds.table.iloc[idx1][train_ds.cols['filename']],
            "label_1": train_ds.table.iloc[idx1][train_ds.cols['label']],
            "image_id_2": train_ds.table.iloc[idx2][train_ds.cols['image_id']],
            "timestamp_2": train_ds.table.iloc[idx2][train_ds.cols['timestamp']],
            "filename_2": train_ds.table.iloc[idx2][train_ds.cols['filename']],
            "label_2": train_ds.table.iloc[idx2][train_ds.cols['label']],
            "pair_label": label,
        }
        for idx1, idx2, label in train_ds.ranked_image_pairs
    ]
    train_pairs_df = pd.DataFrame(train_pairs)
    train_pairs_df.to_csv(os.path.join(output_dir, "pairs-train.csv"), index=False)
    train_filenames = train_pairs_df['filename_1'].tolist() + train_pairs_df['filename_2'].tolist()

    val_pairs = [
        {
            "image_id_1": val_ds.table.iloc[idx1][val_ds.cols['image_id']],
            "timestamp_1": val_ds.table.iloc[idx1][val_ds.cols['timestamp']],
            "filename_1": val_ds.table.iloc[idx1][val_ds.cols['filename']],
            "label_1": val_ds.table.iloc[idx1][val_ds.cols['label']],
            "image_id_2": val_ds.table.iloc[idx2][val_ds.cols['image_id']],
            "timestamp_2": val_ds.table.iloc[idx2][val_ds.cols['timestamp']],
            "filename_2": val_ds.table.iloc[idx2][val_ds.cols['filename']],
            "label_2": val_ds.table.iloc[idx2][val_ds.cols['label']],
            "pair_label": label,
        }
        for idx1, idx2, label in val_ds.ranked_image_pairs
    ]
    val_pairs_df = pd.DataFrame(val_pairs)
    val_pairs_df.to_csv(os.path.join(output_dir, "pairs-val.csv"), index=False)
    val_filenames = val_pairs_df['filename_1'].tolist() + val_pairs_df['filename_2'].tolist()

    test_pairs = [
        {
            "image_id_1": test_ds.table.iloc[idx1][test_ds.cols['image_id']],
            "timestamp_1": test_ds.table.iloc[idx1][test_ds.cols['timestamp']],
            "filename_1": test_ds.table.iloc[idx1][test_ds.cols['filename']],
            "label_1": test_ds.table.iloc[idx1][test_ds.cols['label']],
            "image_id_2": test_ds.table.iloc[idx2][test_ds.cols['image_id']],
            "timestamp_2": test_ds.table.iloc[idx2][test_ds.cols['timestamp']],
            "filename_2": test_ds.table.iloc[idx2][test_ds.cols['filename']],
            "label_2": test_ds.table.iloc[idx2][test_ds.cols['label']],
            "pair_label": label,
        }
        for idx1, idx2, label in test_ds.ranked_image_pairs
    ]
    test_pairs_df = pd.DataFrame(test_pairs)
    test_pairs_df.to_csv(os.path.join(output_dir, "pairs-test.csv"), index=False)
    test_filenames = test_pairs_df['filename_1'].tolist() + test_pairs_df['filename_2'].tolist()

    manifest_filename = os.path.join(output_dir, "manifest.json")
    manifest = train_filenames + val_filenames + test_filenames
    manifest = list(set(manifest)) # unique filenames
    manifest.insert(0, {"prefix": f"s3://{args.images_bucket}/"})
    print(f"writing manifest to {manifest_filename} (num images: {len(manifest)})")
    with open(manifest_filename, "w") as f:
        json.dump(manifest, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--random-seed", type=int, default=1691, help="random seed")
    parser.add_argument("--output-dir", type=str, help="output dir")
    parser.add_argument(
        "--data-file",
        type=str,
        default="flow-images.csv",
        help="filename of CSV file with linked images and flows",
    )

    parser.add_argument(
        "--margin-mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help="type of comparison made by simulated oracle makes of flows in a pair of images",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="max discernable difference in flows between a pair of images",
    )

    parser.add_argument(
        "--num-train-pairs",
        type=int,
        default=5000,
        help="number of labeled image pairs on which to train model",
    )
    parser.add_argument(
        "--num-eval-pairs",
        type=int,
        default=1000,
        help="number of labeled image pairs on which to evaluate model",
    )

    parser.add_argument(
        "--images-bucket",
        type=str,
        help="S3 bucket where images are stored",
    )

    generate_datasets(parser.parse_args())
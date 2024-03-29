import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from functools import reduce

# from torch.utils.data import DataLoader
from torchvision.transforms import (
    Resize,
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip,
    RandomRotation,
    ColorJitter,
    Normalize,
    Compose,
)
import sys

sys.path.append("../")
from src.datasets import (
    RandomStratifiedWeeklyFlow,
    FlowPhotoDataset,
    random_pairs,
)
from src.modules import ResNetRankNet
from src.utils import filter_by_hour, filter_by_month, fit, validate

# import arguments
from tqdm import tqdm


# DATALOADING OPTS
def data_args(parser):
    group = parser.add_argument_group(
        "Data", "Arguments control Data and loading for training"
    )
    group.add_argument(
        "--site",
        type=str,
        required=True,
        help="name of site with linked images and flows",
    )
    group.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="path to CSV file with linked images and flows",
    )
    group.add_argument(
        "--image-root-dir",
        type=str,
        required=True,
        help="path to folder containing images listed in data-file",
    )
    group.add_argument(
        "--col-timestamp",
        type=str,
        default="timestamp",
        help="datetime column name in data-file",
    )
    group.add_argument(
        "--min-hour",
        type=int,
        default=0,
        help="minimum timestamp hour for including samples in data-file",
    )
    group.add_argument(
        "--max-hour",
        type=int,
        default=23,
        help="maximum timestamp hour for including samples in data-file",
    )
    group.add_argument(
        "--min-month",
        type=int,
        default=1,
        help="minimum timestamp month for including samples in data-file",
    )
    group.add_argument(
        "--max-month",
        type=int,
        default=12,
        help="maximum timestamp month for including samples in data-file",
    )
    # group.add_argument(
    #     "--split-idx",
    #     type=int,
    #     required=True,
    #     help="index specifying which of 5 train/val splits to use",
    # )
    group.add_argument(
        "--normalize",
        type=bool,
        default=True,
        help="whether to normalize image inputs to model",
    )
    group.add_argument(
        "--augment",
        type=bool,
        default=True,
        help="whether to use image augmentation during training",
    )
    # group.add_argument(
    #     "--crop-to-bbox",
    #     type=bool,
    #     default=False,
    #     help="whether to crop images to bounding boxes before training",
    # )
    group.add_argument(
        "--batch-size", type=int, default=64, help="batch size of the train loader"
    )
    group.add_argument(
        "--test-batch-size", type=int, default=64, help="batch size of the test loader"
    )


# MODEL OPTS
def model_args(parser):
    group = parser.add_argument_group("Model", "Arguments control Model")
    # group.add_argument('--arch', default='ResNet18', type=str, choices=['ResNet18', 'ResNet50'],
    #                    help='model architecture')
    group.add_argument(
        "--truncate",
        default=2,
        type=int,
        help="number of final layers of model to remove",
    )


# BASE TRAINING ARGS
def base_train_args(parser):
    group = parser.add_argument_group(
        "Base Training", "Base arguments to configure training"
    )
    group.add_argument(
        "--epochs", type=int, default=15, help="number of training epochs"
    )
    group.add_argument("--lr", type=float, default=0.001, help="learning rate")
    group.add_argument(
        "--unfreeze-after",
        type=int,
        default=2,
        help="number of epochs after which to unfreeze model backbone",
    )


# RANKING MODEL DATA ARGS
def ranking_data_args(parser):
    group = parser.add_argument_group(
        "RankNet Training", "Arguments to configure RankNet training data"
    )
    group.add_argument(
        "--margin-mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help="type of comparison made by simulated oracle makes of flows in a pair of images",
    )
    group.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="minimum difference in a pair of streamflow images needed to rank one higher than the other",
    )
    group.add_argument(
        "--num-train-pairs",
        type=int,
        default=5000,
        help="number of labeled image pairs on which to train model",
    )
    group.add_argument(
        "--num-eval-pairs",
        type=int,
        default=1000,
        help="number of labeled image pairs on which to evaluate model",
    )


# # SAVING ARGS
def saving_args(parser):
    group = parser.add_argument_group(
        "Results Saving", "Arguments to configure saving outputs"
    )
    group.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="directory in which to save model checkpoints and metric logs",
    )


# CHECKPOINT LOADING ARGS
def ckpt_loading_args(parser):
    group = parser.add_argument_group(
        "Results Loading", "Arguments to configure loading outputs"
    )
    group.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="path to saved checkpoint to load model weights",
    )


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", default=939, type=int, help="random seed")
    data_args(parser)
    ranking_data_args(parser)
    model_args(parser)
    # base_train_args(parser)
    saving_args(parser)

    # # add temporary arguments that should be refactored out
    # parser.add_argument("--pii-detection-results", default=None)
    args = parser.parse_args()
    return args


def load_data(data_file, col_timestamp="timestamp"):
    df = pd.read_csv(data_file)
    df[col_timestamp] = pd.to_datetime(df[col_timestamp])
    df.sort_values(by=col_timestamp, inplace=True, ignore_index=True)
    return df


def create_image_transforms(
    resize_shape,
    input_shape,
    augmentation=True,
    normalization=True,
    means=None,
    stds=None,
):
    image_transforms = {
        "train": [
            Resize(resize_shape),
        ],
        "eval": [
            Resize(resize_shape),
        ],
    }

    # augmentation
    image_transforms["train"].extend(
        [
            RandomCrop(input_shape),
            RandomHorizontalFlip(),
            RandomRotation(10),
            ColorJitter(),
        ]  # type: ignore
    ) if augmentation else image_transforms["train"].append(
        CenterCrop(input_shape)  # type: ignore
    )  # type: ignore
    image_transforms["eval"].append(CenterCrop(input_shape))  # type: ignore

    # normalization
    if normalization:
        image_transforms["train"].append(Normalize(means, stds))  # type: ignore
        image_transforms["eval"].append(Normalize(means, stds))  # type: ignore

    # composition
    image_transforms["train"] = Compose(image_transforms["train"])  # type: ignore
    image_transforms["eval"] = Compose(image_transforms["eval"])  # type: ignore
    return image_transforms


def inference_ranknet(args):
    print(args)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # REPRODUCIBILITY
    # # # # # # # # # # # # # # # # # # # # # # # # #
    random.seed(args.seed)
    np.random.seed(args.seed)

    # In general seed PyTorch operations
    torch.manual_seed(args.seed)
    # If you are using CUDA on 1 GPU, seed it
    torch.cuda.manual_seed(args.seed)
    # If you are using CUDA on more than 1 GPU, seed them all
    torch.cuda.manual_seed_all(args.seed)
    # Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware.
    # torch.backends.cudnn.benchmark = False # this might be slowing down training
    # Certain operations in Cudnn are not deterministic, and this line will force them to behave!
    # torch.backends.cudnn.deterministic = True # this might be slowing down training

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # LOAD DATASET
    # # # # # # # # # # # # # # # # # # # # # # # # #
    df = load_data(args.data_file)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # FILTER BY TIME OF DAY, MONTH OF YEAR, ETC.
    # # # # # # # # # # # # # # # # # # # # # # # # #
    df_filters = [
        (filter_by_hour, {"min_hour": args.min_hour, "max_hour": args.max_hour}),
        (filter_by_month, {"min_month": args.min_month, "max_month": args.max_month}),
    ]
    df = reduce(lambda _df, filter: _df.pipe(filter[0], **filter[1]), df_filters, df)
    df.reset_index(inplace=True)
    splits = RandomStratifiedWeeklyFlow().split(df, 0.8, 0.1, 0.1)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # CREATE PYTORCH DATASETS AND DATALOADERS
    # # # # # # # # # # # # # # # # # # # # # # # # #
    train_ds = FlowPhotoDataset(train_df, os.path.dirname(args.image_root_dir))
    img_sample_mean, img_sample_std = train_ds.compute_mean_std()
    image = train_ds.get_image(0)
    aspect = image.shape[2] / image.shape[1]
    resize_shape = [480, np.int32(480 * aspect)]
    input_shape = [384, np.int32(384 * aspect)]
    image_transforms = create_image_transforms(
        resize_shape, input_shape, means=img_sample_mean, stds=img_sample_std
    )

    train_ds.transform = image_transforms["eval"]
    val_ds = FlowPhotoDataset(
        val_df, os.path.dirname(args.image_root_dir), transform=image_transforms["eval"]
    )
    test_ds = FlowPhotoDataset(
        test_df,
        os.path.dirname(args.image_root_dir),
        transform=image_transforms["eval"],
    )

    # train_ds.rank_image_pairs(
    #     random_pairs, args.num_train_pairs, args.margin, args.margin_mode
    # )
    # val_ds.rank_image_pairs(
    #     random_pairs, args.num_eval_pairs, args.margin, args.margin_mode
    # )
    # test_ds.rank_image_pairs(
    #     random_pairs, args.num_eval_pairs, args.margin, args.margin_mode
    # )

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=4
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=4
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=4
    )
    print("DATALOADERS CREATED!")
    # # # # # # # # # # # # # # # # # # # # # # # # #
    # INITIALIZE MODEL
    # # # # # # # # # # # # # # # # # # # # # # # # #
    if torch.cuda.is_available():
        use_gpu = True
        device = torch.device("cuda")
        print(f"Using {torch.cuda.device_count()} GPU(s) for eval.")
    else:
        use_gpu = False
        device = torch.device("cpu")
        print("Using CPU for eval.")
    model = ResNetRankNet(
        input_shape=(3, input_shape[0], input_shape[1]),
        resnet_size=18,
        truncate=args.truncate,
        pretrained=True,
    )
    model = torch.nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("MODEL LOADED!")

    paramstrings = []
    paramstrings.append("ranking")
    paramstrings.extend(["margin", str(args.margin)])
    paramstrings.extend(["randompairs", str(args.num_train_pairs)])
    paramstrings.append(args.site)
    if args.augment:
        paramstrings.append("augment")
    if args.normalize:
        paramstrings.append("normalize")
    paramstrings.append(str(args.seed))
    # paramstrings.append(str(args["split_idx"]))
    paramstr = "_".join(paramstrings)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # EVAL
    # # # # # # # # # # # # # # # # # # # # # # # # #
    for dl in [train_dl, val_dl, test_dl]:
        scores = np.empty((len(dl.dataset),))
        sidx = 0
        with torch.no_grad():
            for bidx, batch in tqdm(enumerate(dl), total=len(dl)):
                inputs, labels = batch
                nsamples = labels.shape[0]
                outputs = model.module.forward_single(inputs.to(device))
                scores[sidx : sidx + nsamples] = outputs.detach().cpu().numpy()
                sidx += nsamples
        dl.dataset.table.loc[:, "scores"] = scores

    train_pred_f = "pred_" + paramstr + "_train.csv"
    train_pred_save_path = os.path.join(args.save_dir, train_pred_f)
    train_dl.dataset.table.to_csv(train_pred_save_path)

    val_pred_f = "pred_" + paramstr + "_val.csv"
    val_pred_save_path = os.path.join(args.save_dir, val_pred_f)
    val_dl.dataset.table.to_csv(val_pred_save_path)

    test_pred_f = "pred_" + paramstr + "_test.csv"
    test_pred_save_path = os.path.join(args.save_dir, test_pred_f)
    test_dl.dataset.table.to_csv(test_pred_save_path)

    return train_dl.dataset.table, val_dl.dataset.table, test_dl.dataset.table


if __name__ == "__main__":
    # get args
    args = get_args()
    print(args)

    # convert argparse.Namespace to dictionary: vars(args)
    inference_ranknet(args)

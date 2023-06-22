import os
import sys
import time
import random
import argparse
import copy
import pickle
import json
import pandas as pd
import numpy as np
from functools import reduce
from tqdm import tqdm
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
import torch.utils.data

sys.path.append("../")
print(sys.path)
from src.datasets import (
    RandomStratifiedWeeklyFlow,
    FlowPhotoDataset,
    FlowPhotoRankingDataset,
    random_pairs,
)
from src.modules import ResNetRankNet
from src.losses import RankNetLoss
from src.utils import filter_by_hour, filter_by_month, fit, validate


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
        "--test-batch-size", type=int, default=64, help="batch size of the train loader"
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


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", default=939, type=int, help="random seed")
    data_args(parser)
    ranking_data_args(parser)
    model_args(parser)
    base_train_args(parser)
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


# def filter_detections(detection_results, confidence_threshold: float, categories=[]):
#     """Filter detections by confidence threshold and category.

#     Args:
#         detection_results: A dict containing MegaDetector v5 results.
#         confidence_threshold: A float representing the confidence below
#           which detections should be filtered out.
#         categories: A list of categories of detections to return. Detections
#           in other categories will be filtered out.

#     Returns:
#         A dict containing only MegaDetector v5 detection results above the
#         specified confidence threshold and belonging to the specified
#         categories.

#     Raises:

#     """
#     filtered_results = copy.deepcopy(detection_results)
#     for image in tqdm(filtered_results["images"]):
#         # keep only detections above confidence_threshold
#         # and of the specified categories
#         image["detections"] = [
#             det
#             for det in image["detections"]
#             if (det["conf"] >= confidence_threshold) and (det["category"] in categories)
#         ]
#         image["max_detection_conf"] = (
#             max([det["conf"] for det in image["detections"]])
#             if len(image["detections"]) > 0
#             else 0.0
#         )

#     # keep only images that have at least 1 detection after filtering
#     filtered_results["images"] = [
#         image for image in filtered_results["images"] if len(image["detections"]) > 0
#     ]
#     return filtered_results


# def _load_data_file(filepath, pii_detections):
#     # logger.info(f"load dataset: {filepath}")
#     df = pd.read_csv(filepath, dtype={"flow_cfs": np.float32})
#     df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(tz="US/Eastern")
#     df.sort_values(by="timestamp", inplace=True, ignore_index=True)

#     # filter by hour
#     min_hour = 6
#     max_hour = 18
#     # logger.info(f"filter(hour): {min_hour} to {max_hour}")
#     df = df[df["timestamp"].dt.hour.between(min_hour, max_hour)]

#     # filter by month
#     min_month = 3
#     max_month = 11
#     # logger.info(f"filter(month): {min_month} to {max_month}")
#     df = df[df["timestamp"].dt.month.between(min_month, max_month)]

#     # filter by pii detections
#     pii_results = json.load(open(args.pii_detection_results, "r"))
#     pii_detections = filter_detections(pii_results, 0.2, ["2", "3"])["images"]
#     pii_files = pd.DataFrame(pii_detections)["file"].tolist()
#     df = df[~df["filename"].isin(pii_files)]

#     # logger.info(
#     #     f"dataset loaded\n  rows: {len(df)}\n  flow: {df.flow_cfs.mean():>.2f} cfs"
#     # )
#     return df


if __name__ == "__main__":
    args = get_args()

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
    # LOAD DATA
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
    train_ds = FlowPhotoRankingDataset(train_df, os.path.dirname(args.image_root_dir))
    img_sample_mean, img_sample_std = train_ds.compute_mean_std()
    image = train_ds.get_image(0)
    aspect = image.shape[2] / image.shape[1]
    resize_shape = [480, np.int32(480 * aspect)]
    input_shape = [384, np.int32(384 * aspect)]
    image_transforms = create_image_transforms(
        resize_shape, input_shape, means=img_sample_mean, stds=img_sample_std
    )

    train_ds.transform = image_transforms["train"]
    val_ds = FlowPhotoRankingDataset(
        val_df, os.path.dirname(args.image_root_dir), transform=image_transforms["eval"]
    )
    test_ds = FlowPhotoRankingDataset(
        test_df,
        os.path.dirname(args.image_root_dir),
        transform=image_transforms["eval"],
    )

    train_ds.rank_image_pairs(
        random_pairs, args.num_train_pairs, args.margin, args.margin_mode
    )
    val_ds.rank_image_pairs(
        random_pairs, args.num_eval_pairs, args.margin, args.margin_mode
    )
    test_ds.rank_image_pairs(
        random_pairs, args.num_eval_pairs, args.margin, args.margin_mode
    )

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=4
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=4
    )

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # INITIALIZE MODEL
    # # # # # # # # # # # # # # # # # # # # # # # # #
    if torch.cuda.is_available():
        use_gpu = True
        device = torch.device("cuda")
        print(f"Using {torch.cuda.device_count()} GPU(s) to train.")
    else:
        use_gpu = False
        device = torch.device("cpu")
        print("Using CPU to train.")
    model = ResNetRankNet(
        input_shape=(3, input_shape[0], input_shape[1]),
        resnet_size=18,
        truncate=args.truncate,
        pretrained=True,
    )
    # freeze the resnet backbone
    for p in list(model.children())[0].parameters():
        p.requires_grad = False
    unfreeze_after = args.unfreeze_after
    model = torch.nn.DataParallel(model)
    model.to(device)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # INITIALIZE LOSS, OPTIMIZER, LR SCHEDULER
    # # # # # # # # # # # # # # # # # # # # # # # # #
    criterion = RankNetLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=1, factor=0.5
    )

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # INITIALIZE LOSS LOGS
    # # # # # # # # # # # # # # # # # # # # # # # # #
    metriclogs = {}
    metriclogs["training_loss"] = []
    metriclogs["val_loss"] = []
    metriclogs["test_loss"] = []

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
    # TRAIN
    # # # # # # # # # # # # # # # # # # # # # # # # #
    for epoch in range(0, args.epochs):
        # train
        start_time = time.time()
        avg_loss_training = fit(
            model, criterion, optimizer, train_dl, device, epoch_num=epoch
        )
        stop_time = time.time()
        print("training epoch took %0.1f s" % (stop_time - start_time))
        metriclogs["training_loss"].append(avg_loss_training)

        # validate on val set
        start_time = time.time()
        valset_eval = validate(model, [criterion], val_dl, device)
        stop_time = time.time()
        print("valset eval took %0.1f s" % (stop_time - start_time))
        metriclogs["val_loss"].append(valset_eval[0])

        # validate on test set (peeking)
        start_time = time.time()
        testset_eval = validate(model, [criterion], test_dl, device)
        stop_time = time.time()
        print("testset eval took %0.1f s" % (stop_time - start_time))
        metriclogs["test_loss"].append(testset_eval[0])

        # update lr scheduler
        scheduler.step(valset_eval[0])

        # periodically save model checkpoints
        epoch_checkpoint_file = "./epoch%d_" % epoch + paramstr + ".ckpt"
        epoch_checkpoint_save_path = os.path.join(
            args.save_dir, "checkpoints", epoch_checkpoint_file
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_loss": avg_loss_training,
            },
            epoch_checkpoint_save_path,
        )

        #  after [unfreeze_after] epochs, unfreeze the pretrained body network parameters
        if (epoch + 1) == args.unfreeze_after:
            print("UNFREEZING CNN BODY")
            for p in list(model.children())[0].parameters():
                p.requires_grad = True

    # save losses and any other metrics tracked during training
    metrics_file = "metrics_per_epoch_" + paramstr + ".pkl"
    metrics_save_path = os.path.join(args.save_dir, metrics_file)
    with open(metrics_save_path, "wb") as f:
        pickle.dump(metriclogs, f, protocol=pickle.HIGHEST_PROTOCOL)

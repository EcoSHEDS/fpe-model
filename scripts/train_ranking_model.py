import os
import sys
import time
import json
import pickle
from functools import reduce

import configargparse
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.utils.data
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

PROJECT_ROOT = os.path.abspath(os.path.join(sys.path[0], os.pardir))
sys.path.append(PROJECT_ROOT)
from src.arguments import add_data_args, add_ranking_data_args, add_model_training_args
from src.utils import (
    log,
    next_path,
    set_seeds,
    load_data,
    filter_by_hour,
    filter_by_month,
    fit,
    validate,
)
from src.datasets import (
    RandomStratifiedWeeklyFlow,
    FlowPhotoRankingDataset,
    random_pairs,
)
from src.modules import ResNetRankNet
from src.losses import RankNetLoss


def get_args():
    parser = configargparse.ArgParser()
    parser.add_argument(
        "-c", "--config-file", is_config_file=True, help="config file path"
    )
    parser.add_argument(
        "-o",
        "--output-root-dir",
        required=True,
        help="path to root folder containing outputs from running this script",
    )
    parser.add_argument("-s", "--random-seed", default=1691, help="random seed")
    add_data_args(parser)
    add_ranking_data_args(parser)
    add_model_training_args(parser)
    args = parser.parse_args()
    args.parser = parser
    return args


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


def train_ranking_model(args):
    with open(os.path.join(args.exp_dir, "params.txt"), "w") as f:
        f.write(args.parser.format_values())
    args.logger.info(
        f'Run parameters saved to {os.path.join(args.exp_dir, "params.txt")}'
    )

    set_seeds(args.random_seed)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # CREATE / LOAD DATA SPLITS
    # # # # # # # # # # # # # # # # # # # # # # # # #
    df = load_data(args.data_file)
    # filter by time of day, month of year, etc.
    # https://stackoverflow.com/a/68652715
    df_filters = [
        (filter_by_hour, {"min_hour": args.min_hour, "max_hour": args.max_hour}),
        (filter_by_month, {"min_month": args.min_month, "max_month": args.max_month}),
    ]
    df = reduce(lambda _df, filter: _df.pipe(filter[0], **filter[1]), df_filters, df)
    df.reset_index(inplace=True)
    splits = RandomStratifiedWeeklyFlow().split(df, 0.8, 0.1, 0.1)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    # save the train/val/test splits
    train_df.to_csv(os.path.join(args.exp_dir, "train_data.csv"))
    args.logger.info(
        f'Train split saved to {os.path.join(args.exp_dir, "train_data.csv")}'
    )
    val_df.to_csv(os.path.join(args.exp_dir, "val_data.csv"))
    args.logger.info(f'Val split saved to {os.path.join(args.exp_dir, "val_data.csv")}')
    test_df.to_csv(os.path.join(args.exp_dir, "test_data.csv"))
    args.logger.info(
        f'Test split saved to {os.path.join(args.exp_dir, "test_data.csv")}'
    )

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # CREATE PYTORCH DATASETS AND DATALOADERS
    # # # # # # # # # # # # # # # # # # # # # # # # #
    train_ds = FlowPhotoRankingDataset(train_df, args.image_root_dir)
    # get dataset image means, std, and aspect ratio
    img_sample_mean, img_sample_std = train_ds.compute_mean_std()
    args.logger.info(f"Computed image channelwise means: {img_sample_mean}")
    args.logger.info(f"Computed image channelwise stdevs: {img_sample_std}")
    image = train_ds.get_image(0)
    aspect = image.shape[2] / image.shape[1]
    # set up image transforms
    resize_shape = [480, np.int32(480 * aspect)]
    input_shape = [384, np.int32(384 * aspect)]
    image_transforms = create_image_transforms(
        resize_shape,
        input_shape,
        means=img_sample_mean,
        stds=img_sample_std,
        augmentation=args.augment,
        normalization=args.normalize,
    )
    train_ds.transform = image_transforms["train"]
    val_ds = FlowPhotoRankingDataset(
        val_df,
        args.image_root_dir,
        transform=image_transforms["eval"],
    )
    test_ds = FlowPhotoRankingDataset(
        test_df,
        args.image_root_dir,
        transform=image_transforms["eval"],
    )
    # create ranked image pairs from ground truth
    train_ds.rank_image_pairs(
        random_pairs, args.num_train_pairs, args.margin, args.margin_mode
    )
    val_ds.rank_image_pairs(
        random_pairs, args.num_eval_pairs, args.margin, args.margin_mode
    )
    test_ds.rank_image_pairs(
        random_pairs, args.num_eval_pairs, args.margin, args.margin_mode
    )
    # save the train/val/test ranked image pairs
    train_pairs = [
        {
            "idx1": idx1,
            "idx2": idx2,
            "fn1": train_ds.table.iloc[idx1][train_ds.col_filename],
            "fn2": train_ds.table.iloc[idx2][train_ds.col_filename],
            "label": label,
        }
        for idx1, idx2, label in train_ds.ranked_image_pairs
    ]
    pd.DataFrame(train_pairs).to_csv(os.path.join(args.exp_dir, "train_pairs.csv"))
    args.logger.info(
        f'Train pairs saved to {os.path.join(args.exp_dir, "train_pairs.csv")}'
    )
    val_pairs = [
        {
            "idx1": idx1,
            "idx2": idx2,
            "fn1": val_ds.table.iloc[idx1][val_ds.col_filename],
            "fn2": val_ds.table.iloc[idx2][val_ds.col_filename],
            "label": label,
        }
        for idx1, idx2, label in val_ds.ranked_image_pairs
    ]
    pd.DataFrame(val_pairs).to_csv(os.path.join(args.exp_dir, "val_pairs.csv"))
    args.logger.info(
        f'Val pairs saved to {os.path.join(args.exp_dir, "val_pairs.csv")}'
    )
    test_pairs = [
        {
            "idx1": idx1,
            "idx2": idx2,
            "fn1": test_ds.table.iloc[idx1][test_ds.col_filename],
            "fn2": test_ds.table.iloc[idx2][test_ds.col_filename],
            "label": label,
        }
        for idx1, idx2, label in test_ds.ranked_image_pairs
    ]
    pd.DataFrame(test_pairs).to_csv(os.path.join(args.exp_dir, "test_pairs.csv"))
    args.logger.info(
        f'Test pairs saved to {os.path.join(args.exp_dir, "test_pairs.csv")}'
    )

    # TODO: do we need worker_init_fn here for reproducibility?
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # INITIALIZE MODEL, LOSS, AND OPTIMIZER
    # # # # # # # # # # # # # # # # # # # # # # # # #
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU {args.gpu} to train.")
    else:
        device = torch.device("cpu")
        print("Using CPU to train.")
    model = ResNetRankNet(
        input_shape=(3, input_shape[0], input_shape[1]),
        resnet_size=18,
        truncate=2,
        pretrained=True,
    )
    model = torch.nn.DataParallel(
        model,
        device_ids=[
            args.gpu,
        ],
    )
    model.to(device)
    criterion = RankNetLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # LOAD CHECKPOINT IF RESUMING TRAINING/FINE-TUNING
    # # # # # # # # # # # # # # # # # # # # # # # # #
    starting_epoch = 0
    if args.resume_from_checkpoint or args.warm_start_from_checkpoint:
        if args.resume_from_checkpoint:
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        elif args.warm_start_from_checkpoint:
            checkpoint = torch.load(
                args.warm_start_from_checkpoint, map_location=device
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.resume_from_checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            starting_epoch = checkpoint["epoch"] + 1
        if args.resume_from_checkpoint:
            args.logger.info(f"Loaded model from {args.resume_from_checkpoint}")
        elif args.warm_start_from_checkpoint:
            args.logger.info(
                f"Loaded model from {args.warm_start_from_checkpoint} for warm start"
            )

    unfreeze_after = args.unfreeze_after
    if starting_epoch < unfreeze_after:
        # freeze the resnet backbone
        for p in list(model.module.children())[0].parameters():
            p.requires_grad = False

    # assert False, "trying to figure out what is wrong"

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # INITIALIZE LR SCHEDULER WITH OPTIMIZER
    # # # # # # # # # # # # # # # # # # # # # # # # #
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
    paramstrings.append(str(args.random_seed))
    paramstr = "_".join(paramstrings)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # TRAIN
    # # # # # # # # # # # # # # # # # # # # # # # # #
    for epoch in range(starting_epoch, args.epochs):
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

        # periodically save model checkpoints and metrics
        epoch_checkpoint_file = "./epoch%d_" % epoch + paramstr + ".ckpt"
        epoch_checkpoint_save_path = os.path.join(
            args.exp_dir, "checkpoints", epoch_checkpoint_file
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
        metrics_checkpoint_file = "metrics_per_epoch_" + paramstr + ".json"
        metrics_checkpoint_save_path = os.path.join(
            args.exp_dir, metrics_checkpoint_file
        )
        with open(metrics_checkpoint_save_path, "w") as f:
            json.dump(metriclogs, f)

        #  after [unfreeze_after] epochs, unfreeze the pretrained body network parameters
        if (epoch + 1) == args.unfreeze_after:
            print("UNFREEZING CNN BODY")
            for p in list(model.children())[0].parameters():
                p.requires_grad = True

    # save losses and any other metrics tracked during training
    metrics_file = "metrics_per_epoch_" + paramstr + ".pkl"
    metrics_save_path = os.path.join(args.exp_dir, metrics_file)
    with open(metrics_save_path, "wb") as f:
        pickle.dump(metriclogs, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = get_args()

    # set up output folder for current run
    exp_dirname = "_".join([os.path.splitext(os.path.basename(__file__))[0], args.site])
    args.exp_dir = next_path(os.path.join(args.output_root_dir, f"{exp_dirname}_%s"))
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, "checkpoints"), exist_ok=True)

    # set up logging
    run_logger = log(log_file=os.path.join(args.exp_dir, "run.logs"))
    args.logger = run_logger

    # train
    train_ranking_model(args)

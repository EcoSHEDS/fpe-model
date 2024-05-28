import argparse
import ast
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.utils.data
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    Grayscale,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
)

from datasets import FPERankingPairsDataset
from losses import RankNetLoss
from modules import ResNetRankNet
from utils import fit, load_pairs, set_seeds, validate


def list_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))


def create_image_transforms(
    resize_shape,
    input_shape,
    decolorize=False,
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

    # decolorize
    if decolorize:
        image_transforms["train"].append(Grayscale(num_output_channels=3))
        image_transforms["eval"].append(Grayscale(num_output_channels=3))

    # augmentation
    (
        image_transforms["train"].extend(
            [
                RandomCrop(input_shape),
                RandomHorizontalFlip(),
                RandomRotation(10),
                ColorJitter(),
            ]
        )
        if augmentation
        else image_transforms["train"].append(CenterCrop(input_shape))
    )
    image_transforms["eval"].append(CenterCrop(input_shape))

    # normalization (except when decolorizing)
    if normalization and not decolorize:
        image_transforms["train"].append(Normalize(means, stds))
        image_transforms["eval"].append(Normalize(means, stds))

    # composition
    image_transforms["train"] = Compose(image_transforms["train"])
    image_transforms["eval"] = Compose(image_transforms["eval"])
    return image_transforms


def train(args):
    print("train()")
    print("args:")
    print(args.__dict__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: {}".format(device))

    print(f"images_dir: {args.images_dir}")
    # list_all_files(args.images_dir)

    print(f"values_dir: {args.values_dir}")
    # list_all_files(args.values_dir)

    print(f"output_dir: {args.output_dir}")

    output_data_dir = os.path.join(args.output_dir, "data")

    with open(os.path.join(output_data_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f'saved args: {os.path.join(output_data_dir, "args.json")}')

    print(f"model_dir: {args.model_dir}")

    print(f"set seeds ({args.random_seed})")
    set_seeds(args.random_seed)

    # LOAD DATA

    ds = FPERankingPairsDataset(
        args.root_dir,
        args.pair_file,
    )
    train_indices = ds.data[ds.data["split"] == "train"].index
    val_indices = ds.data[ds.data["split"] == "val"].index
    # FIXME: mean/std should probably be computed on all in-distribution images, not just annotated ones
    img_sample_mean, img_sample_std = ds.compute_mean_std(
        indices=train_indices, n=args.num_image_stats
    )
    ds.set_mean_std(img_sample_mean, img_sample_std)
    train_ds = torch.utils.data.Subset(ds, train_indices)
    val_ds = torch.utils.data.Subset(ds, val_indices)

    # print("loading pairs from csv files")
    # pairs_df = load_pairs(os.path.join(args.values_dir, args.pairs_file))
    # train_df = pairs_df[pairs_df["split"] == "train"]
    # print(f"train_df: {train_df.shape[0]} rows")
    # val_df = pairs_df[pairs_df["split"] == "val"]
    # print(f"val_df: {val_df.shape[0]} rows")

    # print("creating train dataset")
    # train_ds = FPERankingPairsDataset(
    #     train_df,
    #     args.images_dir,
    # )

    # print("computing image stats")
    # img_sample_mean, img_sample_std = train_ds.compute_mean_std(args.num_image_stats)
    # print(f"img_sample_mean: {img_sample_mean}")
    # print(f"img_sample_std: {img_sample_std}")

    img_1, _, _ = train_ds[0]
    aspect = img_1.shape[2] / img_1.shape[1]
    resize_shape = [args.input_size, np.int32(args.input_size * aspect)]
    print(f"resize_shape: {resize_shape}")

    input_shape = [
        np.int32(args.input_size * 0.8),
        np.int32(args.input_size * 0.8 * aspect),
    ]
    print(f"input_shape: {input_shape}")

    image_transforms = create_image_transforms(
        resize_shape,
        input_shape,
        means=img_sample_mean,
        stds=img_sample_std,
        decolorize=args.decolorize,
        augmentation=args.augment,
        normalization=args.normalize,
    )
    train_ds.transform = image_transforms["train"]
    val_ds.transform = image_transforms["eval"]

    # print("creating val datasets")
    # print(f"val df: {len(val_df)}")
    # val_ds = FPERankingPairsDataset(
    #     val_df,
    #     args.images_dir,
    #     transform=image_transforms["eval"],
    # )
    # print(f"val ds: {len(val_ds)}")

    print("creating data loaders")
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # INITIALIZE MODEL
    # # # # # # # # # # # # # # # # # # # # # # # # #
    print("initializing model")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"using GPU {args.gpu} to train")
    else:
        device = torch.device("cpu")
        print("using CPU to train")
    model = ResNetRankNet(
        input_shape=(3, input_shape[0], input_shape[1]),
        transforms=image_transforms,
        resnet_size=18,
        truncate=2,
        pretrained=True,
    )

    # freeze resnet backbone
    for p in list(model.children())[0].parameters():
        p.requires_grad = False

    model = torch.nn.DataParallel(
        model,
        device_ids=[
            args.gpu,
        ],
    )
    model.to(device)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # INITIALIZE LOSS, OPTIMIZER, SCHEDULER
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
    metriclogs["epoch"] = []
    metriclogs["train_loss"] = []
    metriclogs["val_loss"] = []

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # TRAIN
    # # # # # # # # # # # # # # # # # # # # # # # # #
    print("start training")
    min_val_loss = None
    for epoch in range(0, args.epochs):
        print(f"start training (epoch={epoch})")

        metriclogs["epoch"].append(epoch)

        start_time = time.time()
        train_loss = fit(model, criterion, optimizer, train_dl, device, epoch_num=epoch)
        stop_time = time.time()
        metriclogs["train_loss"].append(train_loss)
        print("train step took %0.1f s" % (stop_time - start_time))
        print("train loss = %0.2f" % (train_loss))

        # validate on val set
        start_time = time.time()
        val_loss = validate(model, [criterion], val_dl, device)
        stop_time = time.time()
        metriclogs["val_loss"].append(val_loss[0])
        print("val step took %0.1f s" % (stop_time - start_time))
        print("val loss = %0.2f" % (val_loss[0]))
        print(
            f"[Epoch {epoch}|val]\t{(stop_time - start_time):.2f} s\t{(val_loss[0]):.4f}"
        )

        # update lr scheduler
        scheduler.step(val_loss[0])

        # periodically save model checkpoints and metrics
        epoch_checkpoint_file = "epoch_%02d" % epoch + ".pth"
        if args.local:
            epoch_checkpoint_save_path = os.path.join(
                args.model_dir, epoch_checkpoint_file
            )
        else:
            epoch_checkpoint_save_path = os.path.join(
                args.checkpoint_dir, epoch_checkpoint_file
            )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_loss": train_loss,
                "transforms": image_transforms,
                "params": {
                    "aspect": aspect,
                    "input_shape": input_shape,
                    "img_sample_mean": img_sample_mean,
                    "img_sample_std": img_sample_std,
                },
            },
            epoch_checkpoint_save_path,
        )

        if min_val_loss is None or val_loss[0] < min_val_loss:
            if min_val_loss is None:
                print(
                    f"updating final model (epoch={epoch}), first val loss ({val_loss[0]:0.3f})"
                )
            else:
                print(
                    f"updating final model (epoch={epoch}), lowest val loss ({val_loss[0]:0.3f} < {min_val_loss:0.3f})"
                )
            final_model_path = os.path.join(args.model_dir, "model.pth")
            shutil.copy(epoch_checkpoint_save_path, final_model_path)
            min_val_loss = val_loss[0]

        metrics_checkpoint_file = "metrics.csv"
        metrics_checkpoint_save_path = os.path.join(
            output_data_dir, metrics_checkpoint_file
        )
        pd.DataFrame(metriclogs).to_csv(metrics_checkpoint_save_path, index=False)

        #  after [unfreeze_after] epochs, unfreeze the pretrained body network parameters
        if (epoch + 1) == args.unfreeze_after:
            print(f"unfreezing cnn body after epoch={epoch}")
            for p in list(model.children())[0].parameters():
                p.requires_grad = True

    print("finished")


def save_model(model, model_dir):
    print("save model")
    path = os.path.join(model_dir, "model")
    model.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker parameters
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument(
        "--hosts", type=str, default=ast.literal_eval(os.environ["SM_HOSTS"])
    )
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--checkpoint-dir", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--output-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument(
        "--images-dir", type=str, default=os.environ["SM_CHANNEL_IMAGES"]
    )
    parser.add_argument(
        "--values-dir", type=str, default=os.environ["SM_CHANNEL_VALUES"]
    )
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    parser.add_argument(
        "--num-workers", type=int, default=4, help="number of data loader workers"
    )
    parser.add_argument("--gpu", type=int, default=0, help="index of the GPU to use")
    parser.add_argument(
        "--local", type=bool, default=False, help="running in local mode"
    )

    # hyperparameters
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--unfreeze-after",
        type=int,
        default=2,
        help="number of epochs after which to unfreeze model backbone",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="batch size of the train loader"
    )
    parser.add_argument("--random-seed", type=int, default=1691, help="random seed")

    # transforms
    parser.add_argument(
        "--num-image-stats",
        type=int,
        default=1000,
        help="number of images to compute mean/stdev",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=480,
        help="image input size to model",
    )
    parser.add_argument(
        "--decolorize",
        type=bool,
        default=False,
        help="remove image color channels",
    )
    parser.add_argument(
        "--normalize",
        type=bool,
        default=True,
        help="whether to normalize image inputs to model",
    )
    parser.add_argument(
        "--augment",
        type=bool,
        default=True,
        help="whether to use image augmentation during training",
    )

    # input files
    parser.add_argument(
        "--root-dir",
        type=str,
        help="root directory containing images and values directories",
    )
    parser.add_argument(
        "--pairs-file",
        type=str,
        default="pairs.csv",
        help="filename of CSV file with annotated image pairs",
    )

    train(parser.parse_args())

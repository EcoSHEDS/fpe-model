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
    Grayscale,
    Normalize,
    RandomCrop,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
)

from datasets import DatasetSubset, FPERankingPairsDataset
from losses import RankNetLoss
from modules import ResNetRankNet
from utils import (
    ArgumentBuilder,
    TransformBuilder,
    fit,
    load_pairs,
    set_seeds,
    validate,
)


def create_image_transforms(
    resize_shape,
    input_shape,
    decolorize=False,
    augmentation=True,
    normalization=True,
    means=None,
    stds=None,
):
    transform_builder = TransformBuilder()
    transform_builder.add_transforms(
        ["train", "eval"], [(Resize, {"size": resize_shape})]
    )
    if decolorize:
        transform_builder.add_transforms(
            ["train", "eval"], [(Grayscale, {"num_output_channels": 3})]
        )
    if augmentation:
        transform_builder.add_transforms(
            "train",
            [
                (RandomHorizontalFlip, {}),
                (RandomRotation, {"degrees": 10}),
                (RandomCrop, {"size": input_shape}),
                (ColorJitter, {}),
                (RandomGrayscale, {}) if not decolorize else None,
            ],
        )
    else:
        transform_builder.add_transforms("train", [(CenterCrop, {"size": input_shape})])
    transform_builder.add_transforms("eval", [(CenterCrop, {"size": input_shape})])
    if normalization:
        transform_builder.add_transforms(
            ["train", "eval"], [(Normalize, {"mean": means, "std": stds})]
        )
    return transform_builder.build()


def train(args):
    print("train()")
    print("args:")
    print(args.__dict__)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    print(f"images_dir: {args.images_dir}")

    print(f"values_dir: {args.values_dir}")

    print(f"output_dir: {args.output_dir}")

    output_data_dir = os.path.join(args.output_dir, "data")

    with open(os.path.join(output_data_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f'saved args: {os.path.join(output_data_dir, "args.json")}')

    print(f"model_dir: {args.model_dir}")

    print(f"set seeds ({args.random_seed})")
    set_seeds(args.random_seed)

    # LOAD DATA

    ds = FPERankingPairsDataset(args.root_dir, args.pair_file)
    train_indices = ds.data[ds.data["split"] == "train"].index
    val_indices = ds.data[ds.data["split"] == "val"].index
    # FIXME: mean/std should probably be computed on all in-distribution images, not just annotated ones
    img_sample_mean, img_sample_std = ds.compute_mean_std(
        indices=train_indices, n=args.num_image_stats
    )
    ds.set_mean_std(img_sample_mean, img_sample_std)

    # print("loading pairs from csv files")
    # print(f"train_df: {train_df.shape[0]} rows")
    # print(f"val_df: {val_df.shape[0]} rows")

    # print("computing image stats")
    # print(f"img_sample_mean: {img_sample_mean}")
    # print(f"img_sample_std: {img_sample_std}")

    img_1, _, _ = ds[0]
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
    print("creating train dataset")
    train_ds = DatasetSubset(ds, train_indices, transform=image_transforms["train"])

    print("creating val datasets")
    # print(f"val df: {len(val_df)}")
    val_ds = DatasetSubset(ds, val_indices, transform=image_transforms["eval"])
    print(f"val ds: {len(val_ds)}")

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
    arg_builder = ArgumentBuilder()
    parser = (
        arg_builder.add_resource_args()
        .add_hyperparameter_args()
        .add_transform_args()
        .build()
    )

    # SageMaker parameters
    # These are used when the script is run within a SageMaker training job
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

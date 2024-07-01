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
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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

from src.datasets import DatasetSubset, FPERankingPairsDataset

# from src.modules import ResNetRankNet
from src.lightning_modules import LossLoggingCallback, RankNetModule
from src.losses import RankNetLoss
from src.utils import (  # Trainer,
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

    print(f"output_dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    args_save_path = os.path.join(args.output_dir, "args.json")
    with open(args_save_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"saved args: {args_save_path}")

    print(f"model_dir: {args.model_dir}")

    print(f"set seeds ({args.random_seed})")
    set_seeds(args.random_seed)

    # LOAD DATA

    ds = FPERankingPairsDataset(args.images_dir, args.pairs_file)
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
    # print("initializing model")
    # model = ResNetRankNet(
    #     input_shape=(3, input_shape[0], input_shape[1]),
    #     transforms=image_transforms,
    #     resnet_size=18,
    #     truncate=2,
    #     pretrained=True,
    # )

    # # freeze resnet backbone
    # for p in list(model.children())[0].parameters():
    #     p.requires_grad = False

    # model = torch.nn.DataParallel(
    #     model,
    #     device_ids=[
    #         args.gpu,
    #     ],
    # )
    # model.to(device)
    model_params = {
        "input_shape": (3, input_shape[0], input_shape[1]),
        "transforms": image_transforms,
        "resnet_size": 18,
        "truncate": 2,
        "pretrained": True,
    }

    module = RankNetModule(
        model_params,
        RankNetLoss(),
        lr=args.lr,
        freeze_layers=["resnetbody"],
        unfreeze_after=args.unfreeze_after,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    metrics_logging_save_path = os.path.join(args.output_dir, "losses.pkl")
    metrics_logging_callback = LossLoggingCallback(filepath=metrics_logging_save_path)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.model_dir,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,  # Save the model after each epoch
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, metrics_logging_callback, checkpoint_callback],
        devices=[args.gpu],
    )
    trainer.fit(module, train_dl, val_dl)

    # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # INITIALIZE LOSS, OPTIMIZER, SCHEDULER
    # # # # # # # # # # # # # # # # # # # # # # # # # #
    # criterion = RankNetLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, "min", patience=1, factor=0.5
    # )
    # trainer = Trainer(
    #     model=model,
    #     criterion=criterion,
    #     validation_metrics=[criterion],
    #     optimizer=optimizer,
    #     device=device,
    #     scheduler=scheduler,
    #     checkpoint_dir=args.model_dir,
    #     transforms=image_transforms,
    #     transforms_params={
    #         "aspect": aspect,
    #         "input_shape": input_shape,
    #         "img_sample_means": img_sample_mean,
    #         "img_sample_stds": img_sample_std,
    #     },
    # )
    # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # INITIALIZE LOSS LOGS
    # # # # # # # # # # # # # # # # # # # # # # # # # #
    # metriclogs = {}
    # metriclogs["epoch"] = []
    # metriclogs["train_loss"] = []
    # metriclogs["val_loss"] = []

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # TRAIN
    # # # # # # # # # # # # # # # # # # # # # # # # #
    # print("start training")
    # for epoch in range(0, args.epochs):
    # print(f"start training (epoch={epoch})")
    # metriclogs["epoch"].append(epoch)

    # start_time = time.time()
    # train_loss = fit(model, criterion, optimizer, train_dl, device, epoch_num=epoch)
    # stop_time = time.time()
    # train_loss, train_time = trainer.train_one_epoch(train_dl, epoch=epoch)
    # metriclogs["train_loss"].append(train_loss)
    # print("train step took %0.1f s" % train_time)
    # print("train loss = %0.2f" % (train_loss))

    # validate on val set
    # start_time = time.time()
    # val_loss = validate(model, [criterion], val_dl, device)
    # val_metrics, val_time = trainer.validate(val_dl, epoch=epoch)
    # stop_time = time.time()
    # metriclogs["val_loss"].append(val_metrics["RankNetLoss"])
    # print("val step took %0.1f s" % val_time)
    # print("val loss = %0.2f" % (val_metrics["RankNetLoss"]))
    # print(
    #     f"[Epoch {epoch}|val]\t{val_time:.2f} s\t{val_metrics['RankNetLoss']:.4f}"
    # )

    # # update lr scheduler
    # scheduler.step(val_metrics["RankNetLoss"])

    # if min_val_loss is None or val_loss[0] < min_val_loss:
    #     if min_val_loss is None:
    #         print(
    #             f"updating final model (epoch={epoch}), first val loss ({val_loss[0]:0.3f})"
    #         )
    #     else:
    #         print(
    #             f"updating final model (epoch={epoch}), lowest val loss ({val_loss[0]:0.3f} < {min_val_loss:0.3f})"
    #         )
    #     final_model_path = os.path.join(args.model_dir, "model.pth")
    #     shutil.copy(epoch_checkpoint_save_path, final_model_path)
    #     min_val_loss = val_loss[0]

    # metrics_checkpoint_file = "metrics.csv"
    # metrics_checkpoint_save_path = os.path.join(
    #     args.output_dir, metrics_checkpoint_file
    # )
    # pd.DataFrame(metriclogs).to_csv(metrics_checkpoint_save_path, index=False)

    # #  after [unfreeze_after] epochs, unfreeze the pretrained body network parameters
    # if (epoch + 1) == args.unfreeze_after:
    #     print(f"unfreezing cnn body after epoch={epoch}")
    #     for p in list(model.children())[0].parameters():
    #         p.requires_grad = True
    #     # reinitialize the optimizer and scheduler
    #     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, "min", patience=1, factor=0.5
    #     )
    #     trainer.optimizer = optimizer
    #     trainer.scheduler = scheduler

    # print("finished")


if __name__ == "__main__":
    arg_builder = ArgumentBuilder()
    parser = (
        arg_builder.add_path_args()
        .add_resource_args()
        .add_hyperparameter_args()
        .add_transform_args()
        .build()
    )

    # # SageMaker parameters
    # # These are used when the script is run within a SageMaker training job
    # # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    # parser.add_argument(
    #     "--hosts", type=str, default=ast.literal_eval(os.environ["SM_HOSTS"])
    # )
    # parser.add_argument(
    #     "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    # )
    # parser.add_argument("--checkpoint-dir", type=str, default="/opt/ml/checkpoints")
    # parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()
    print(args)

    train(args)

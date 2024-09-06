import ast
import glob
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, loggers
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

from src.datasets import DatasetSubset, FPERankingPairsDataset, FPEDataset

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

def predict(args):
    print("predict()")
    print("args:")
    print(args.__dict__)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    print(f"output_dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"model_dir: {args.model_dir}")

    print(f"set seeds ({args.random_seed})")
    set_seeds(args.random_seed)

    # LOAD DATA

    ds = FPERankingPairsDataset(args.images_dir, args.pairs_file)
    train_idxs = ds.data[ds.data["split"] == "train"].index.tolist()
    val_idxs = ds.data[ds.data["split"] == "val"].index.tolist()
    test_idxs = ds.data[ds.data["split"] == "test"].index.tolist()
    # FIXME: mean/std should probably be computed on all in-distribution images, not just annotated ones
    img_sample_mean, img_sample_std = ds.compute_mean_std(
        indices=train_idxs, n=args.num_image_stats
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

    img_tf = create_image_transforms(
        resize_shape,
        input_shape,
        means=img_sample_mean,
        stds=img_sample_std,
        decolorize=args.decolorize,
        augmentation=args.augment,
        normalization=args.normalize,
    )

    # INITIALIZE MODEL
    model_params = {
        "input_shape": (3, input_shape[0], input_shape[1]),
        "transforms": img_tf,
        "resnet_size": 18,
        "truncate": 2,
        "pretrained": True,
    }

    # module = RankNetModule(
    #     model_params,
    #     RankNetLoss(),
    #     lr=args.lr,
    #     freeze_layers=["resnetbody"],
    #     unfreeze_after=args.unfreeze_after,
    # )

    # Initialize the trainer
    trainer = Trainer(
        devices=[args.gpu],
        logger=False,
    )
    
    # Load the best checkpoint
    checkpoint_files = glob.glob(os.path.join(args.model_dir, "best-checkpoint*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found that starts with 'best-checkpoint' in {}".format(args.model_dir))
    checkpoint_path = checkpoint_files[0]
    print(f"Loading checkpoint: {checkpoint_path}")
    
    module = RankNetModule.load_from_checkpoint(
        checkpoint_path,
        model_args=model_params,
        criterion=RankNetLoss(),
        lr=args.lr,
        freeze_layers=["resnetbody"],
        unfreeze_after=args.unfreeze_after
    )
    module.eval()
    
    # Create the dataset and dataloader for prediction
    predict_ds = FPEDataset(args.images_dir, data_file="images_2024-04-10_WB-Stations.csv", transform=img_tf["eval"])
    print(len(predict_ds))
    predict_dl = DataLoader(
        predict_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Run prediction
    print(f"predicting on {len(predict_ds)} images over {len(predict_dl)} batches")
    predictions = trainer.predict(module, predict_dl)
    predictions = np.concatenate([p.cpu().numpy() for p in predictions])
    predict_ds.data.loc[:, "scores"] = predictions

    # Save predictions
    predictions_save_path = os.path.join(args.output_dir, "predictions.csv")
    pd.DataFrame(predict_ds.data).to_csv(predictions_save_path)
    print(f"saved predictions: {predictions_save_path}")



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
    predict(args)
import os
import sys
import pickle
from argparse import Namespace

import configargparse
import numpy as np
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
from tqdm import tqdm

# import configargparse

PROJECT_ROOT = os.path.abspath(os.path.join(sys.path[0], os.pardir))
sys.path.append(PROJECT_ROOT)
# from src.arguments import add_data_args, add_ranking_data_args
from src.utils import parse_configargparse_args, next_path, log, load_data
from src.datasets import FlowPhotoDataset
from src.modules import ResNetRankNet


def get_args():
    parser = configargparse.ArgParser()
    parser.add_argument(
        "--inference-data-file",
        required=True,
        help="path to csv file containing image paths for inference",
    )
    parser.add_argument(
        "--inference-image-root-dir",
        required=True,
        help="path to folder containing images for inference",
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        help="path to checkpoint file for ranking model",
    )
    parser.add_argument(
        "--inference-output-root-dir",
        required=True,
        help="path to folder where inference results will be saved",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="index of GPU to use for inference",
    )
    args = parser.parse_args()

    # get parameters and values used during checkpointed model training
    args.train_output_dir = os.path.abspath(
        os.path.join(args.ckpt_path, os.pardir, os.pardir)
    )
    params_file = os.path.join(args.train_output_dir, "params.txt")
    training_args = parse_configargparse_args(params_file)

    # override gpu arg from params file
    if "gpu" in args:
        del training_args["gpu"]
    # update param names from training params file
    training_args["train_data_file"] = training_args["data_file"]
    del training_args["data_file"]
    training_args["train_image_root_dir"] = training_args["image_root_dir"]
    del training_args["image_root_dir"]
    training_args["train_site"] = training_args["site"]
    del training_args["site"]
    args = Namespace(**vars(args), **training_args)
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


def inference_ranking_model(args):
    with open(os.path.join(args.exp_dir, "params.pkl"), "wb") as f:
        pickle.dump(vars(args), f)
    args.logger.info(
        f'Run parameters saved to {os.path.join(args.exp_dir, "params.pkl")}'
    )

    df = load_data(args.inference_data_file)
    ds = FlowPhotoDataset(df, args.inference_image_root_dir, col_label=args.col_label)
    image = ds.get_image(0)
    aspect = image.shape[2] / image.shape[1]
    # set up image transforms
    resize_shape = [480, np.int32(480 * aspect)]
    input_shape = [384, np.int32(384 * aspect)]
    image_transforms = create_image_transforms(
        resize_shape,
        input_shape,
        means=args.img_sample_mean,
        stds=args.img_sample_std,
        augmentation=args.augment,
        normalization=args.normalize,
    )
    ds.transform = image_transforms["eval"]  # use eval transforms during inference
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=24
    )

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # LOAD TRAINED MODEL
    # # # # # # # # # # # # # # # # # # # # # # # # #
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU {args.gpu} for inference.")
    else:
        device = torch.device("cpu")
        print("Using CPU for inference¬.")
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
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded model from checkpoint.")

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # RUN INFERENCE
    # # # # # # # # # # # # # # # # # # # # # # # # #
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
    print("Inference complete.")

    # save results
    predictions_save_path = os.path.join(
        args.exp_dir,
        "inference_results_" + os.path.basename(args.inference_data_file),
    )
    dl.dataset.table.to_csv(predictions_save_path)
    args.logger.info(f"Inference results saved to {predictions_save_path}")


if __name__ == "__main__":
    args = get_args()

    exp_dirname = os.path.splitext(os.path.basename(__file__))[0]
    args.exp_dir = next_path(
        os.path.join(args.inference_output_root_dir, f"{exp_dirname}_%s")
    )
    os.makedirs(args.exp_dir, exist_ok=True)

    # set up logging
    run_logger = log(log_file=os.path.join(args.exp_dir, "run.logs"))
    args.logger = run_logger

    # get image sample mean and std from training experiment logs
    exp_log_file = os.path.join(args.train_output_dir, "run.logs")
    with open(exp_log_file, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            if "Computed image channelwise means" in line:
                args.img_sample_mean = np.array(
                    [float(x) for x in line.split(":")[-1].strip()[1:-1].split()]
                )
            if "Computed image channelwise stdevs" in line:
                args.img_sample_std = np.array(
                    [float(x) for x in line.split(":")[-1].strip()[1:-1].split()]
                )

    inference_ranking_model(args)

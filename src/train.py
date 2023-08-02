import argparse
import ast
import os
import time
import torch
import json
import pickle
from functools import reduce
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
from arguments import add_data_args, add_ranking_data_args, add_model_training_args
from utils import (
    log,
    next_path,
    set_seeds,
    load_data,
    filter_by_hour,
    filter_by_month,
    fit,
    validate,
)
from datasets import (
    RandomStratifiedWeeklyFlow,
    FlowPhotoRankingDataset,
    random_pairs,
)
from modules import ResNetRankNet
from losses import RankNetLoss

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def list_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))

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

def train(args):
    print("train()")
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: {}".format(device))

    print(f"images_dir: {args.images_dir}")
    # list_all_files(args.images_dir)
    
    print(f"values_dir: {args.values_dir}")
    # list_all_files(args.values_dir)

    print(f"output_dir: {args.output_dir}")

    output_data_dir = os.path.join(args.output_dir, "data")
    
    with open(os.path.join(output_data_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent = 2)
    print(f'saved args: {os.path.join(output_data_dir, "args.json")}')

    print(f"model_dir: {args.model_dir}")

    set_seeds(args.random_seed)

    print("set seed")
    set_seeds(args.random_seed)

    print("loading data file")
    data_filepath = os.path.join(args.values_dir, args.data_file)
    df = load_data(data_filepath)

    df_filters = [
        (filter_by_hour, {"min_hour": args.min_hour, "max_hour": args.max_hour}),
        (filter_by_month, {"min_month": args.min_month, "max_month": args.max_month}),
    ]
    df = reduce(lambda _df, filter: _df.pipe(filter[0], **filter[1]), df_filters, df)
    df.reset_index(inplace=True)
    splits = RandomStratifiedWeeklyFlow().split(df, 0.8, 0.1, 0.1)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    
    # save the train/val/test splits
    train_df.to_csv(os.path.join(output_data_dir, "train_data.csv"))
    val_df.to_csv(os.path.join(output_data_dir, "val_data.csv"))
    test_df.to_csv(os.path.join(output_data_dir, "test_data.csv"))

    train_ds = FlowPhotoRankingDataset(train_df, args.images_dir)
    img_sample_mean, img_sample_std = train_ds.compute_mean_std(args.num_image_stats)
    print(f"image channelwise means: {img_sample_mean}")
    print(f"image channelwise stdevs: {img_sample_std}")
    
    image = train_ds.get_image(0)
    aspect = image.shape[2] / image.shape[1]
    resize_shape = [480, np.int32(480 * aspect)]
    input_shape = [384, np.int32(384 * aspect)]
    image_transforms = create_image_transforms(
        resize_shape, input_shape, means=img_sample_mean, stds=img_sample_std
    )
    train_ds.transform = image_transforms["train"]
    val_ds = FlowPhotoRankingDataset(
        val_df,
        args.images_dir,
        transform=image_transforms["eval"],
    )
    test_ds = FlowPhotoRankingDataset(
        test_df,
        args.images_dir,
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
    pd.DataFrame(train_pairs).to_csv(os.path.join(output_data_dir, "train_pairs.csv"))
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
    pd.DataFrame(val_pairs).to_csv(os.path.join(output_data_dir, "val_pairs.csv"))
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
    pd.DataFrame(test_pairs).to_csv(os.path.join(output_data_dir, "test_pairs.csv"))

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
    # INITIALIZE MODEL
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
    # freeze the resnet backbone
    for p in list(model.children())[0].parameters():
        p.requires_grad = False
    unfreeze_after = args.unfreeze_after
    model = torch.nn.DataParallel(
        model,
        device_ids=[
            args.gpu,
        ],
    )
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
    paramstrings.append(str(args.random_seed))
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

        # periodically save model checkpoints and metrics
        epoch_checkpoint_file = "./epoch%d_" % epoch + paramstr + ".ckpt"
        epoch_checkpoint_save_path = os.path.join(
            args.checkpoint_dir, epoch_checkpoint_file
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
            output_data_dir, metrics_checkpoint_file
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
    metrics_save_path = os.path.join(output_data_dir, metrics_file)
    with open(metrics_save_path, "wb") as f:
        pickle.dump(metriclogs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save final model
    final_model_path = os.path.join(args.model_dir, "model.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_loss": avg_loss_training,
        },
        final_model_path,
    )
    
    print("finished")


def save_model(model, model_dir):
    print("save model")
    path = os.path.join(model_dir, "model")
    model.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--custom", type=str, default="streamflow", help="dummy custom argument"
    # )

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument("--hosts", type=str, default=ast.literal_eval(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--checkpoint-dir", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--output-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--images-dir", type=str, default=os.environ["SM_CHANNEL_IMAGES"])
    parser.add_argument("--values-dir", type=str, default=os.environ["SM_CHANNEL_VALUES"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    parser.add_argument("--gpu", type=int, default=0, help="index of the GPU to use")
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--unfreeze-after", type=int, default=2,
        help="number of epochs after which to unfreeze model backbone",
    )
    parser.add_argument("--random-seed", type=int, default=1691, help="random seed")
    
    parser.add_argument(
        "--margin-mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help="type of comparison made by simulated oracle makes of flows in a pair of images",
    )
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument(
        "--num-image-stats",
        type=int,
        default=1000,
        help="number of images to compute mean/stdev",
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
        "--site",
        type=str,
        required=True,
        help="name of site with linked images and flows",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="flow-images.csv",
        help="filename of CSV file with linked images and flows",
    )
    parser.add_argument(
        "--col-timestamp",
        type=str,
        default="timestamp",
        help="datetime column name in data-file",
    )
    parser.add_argument(
        "--min-hour",
        type=int,
        default=0,
        help="minimum timestamp hour for including samples in data-file",
    )
    parser.add_argument(
        "--max-hour",
        type=int,
        default=23,
        help="maximum timestamp hour for including samples in data-file",
    )
    parser.add_argument(
        "--min-month",
        type=int,
        default=1,
        help="minimum timestamp month for including samples in data-file",
    )
    parser.add_argument(
        "--max-month",
        type=int,
        default=12,
        help="maximum timestamp month for including samples in data-file",
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
    parser.add_argument(
        "--batch-size", type=int, default=64, help="batch size of the train loader"
    )

    train(parser.parse_args())
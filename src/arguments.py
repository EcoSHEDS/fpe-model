def add_data_args(parser):
    group = parser.add_argument_group(
        "Data", "Arguments for specifying dataset and data loading"
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
    group.add_argument(
        "--batch-size", type=int, default=64, help="batch size of the train loader"
    )


# RANKING MODEL DATA ARGS
def add_ranking_data_args(parser):
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
    group.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="path to CSV file with annotations for ranking model training",
    )


# MODEL TRAINING ARGS
def add_model_training_args(parser):
    group = parser.add_argument_group(
        "Model Training", "Base arguments to configure training"
    )
    group.add_argument("--gpu", type=int, default=0, help="index of the GPU to use")
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
    ckpt_group = group.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="path to checkpoint from which to resume training",
    )
    ckpt_group.add_argument(
        "--warm-start-from-checkpoint",
        type=str,
        default=None,
        help="path to checkpoint from which to warm start training",
    )

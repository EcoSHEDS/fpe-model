import argparse
import ast
import os
import torch
import json
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

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def list_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))

def train(args):
    print("train()")
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: {}".format(device))

    print(f"images_dir: {args.images_dir}")
    list_all_files(args.images_dir)
    
    print(f"values_dir: {args.values_dir}")
    list_all_files(args.values_dir)

    print(f"output_dir: {args.output_dir}")
    
    with open(os.path.join(args.output_dir, "data", "args.json"), "w") as f:
        json.dump(vars(args), f, indent = 2)
    print(f'saved args: {os.path.join(args.output_dir, "data", "args.json")}')

    print(f"model_dir: {args.model_dir}")

    print("finished")


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "my_model")
    model.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--custom", type=str, default="streamflow", help="dummy custom argument"
    )

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument("--hosts", type=str, default=ast.literal_eval(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--output-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--images-dir", type=str, default=os.environ["SM_CHANNEL_IMAGES"])
    parser.add_argument("--values-dir", type=str, default=os.environ["SM_CHANNEL_VALUES"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())
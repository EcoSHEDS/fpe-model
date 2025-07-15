import os
import argparse
import json
import io
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
from modules import ResNetRankNet
from datasets import FlowPhotoDataset
from utils import load_data
import torch

params = None

def model_fn(model_dir):
    print("model_fn()")
    print("model_dir: {}".format(model_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    model_file = os.path.join(model_dir, "model.pth")
    print(f"loading checkpoint: {model_file}")
    with open(model_file, "rb") as f:
        checkpoint = torch.load(f, map_location=device)
        params = checkpoint["params"]
        model = ResNetRankNet(
            input_shape=(3, params["input_shape"][0], params["input_shape"][1]),
            transforms=checkpoint["transforms"],
            resnet_size=18,
            truncate=2,
            pretrained=True,
        )
        model = torch.nn.DataParallel(
            model,
            device_ids=[],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print("model loaded from checkpoint")

    model.eval()
    return model.to(device)

def load_from_bytearray(request_body):
    try:
        image_as_bytes = io.BytesIO(request_body)
        image = Image.open(image_as_bytes)
        image_tensor = ToTensor()(image)
        return image_tensor
    except Exception as e:
        print(f"Error loading image: {e}")
        raise ValueError("Invalid image payload")


def input_fn(request_body, request_content_type):
    # if set content_type as "image/jpg" or "application/x-npy",
    # the input is also a python bytearray
    if request_content_type == "image/jpg":
        image_tensor = load_from_bytearray(request_body)
    else:
        print("not support this type yet")
        raise ValueError("not support this type yet")
    return image_tensor


# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    transformed_object = model.module.transforms['eval'](input_object)
    output = model.module.forward_single(transformed_object.unsqueeze(0))
    pred = output.detach().cpu().numpy()

    return {"score": pred.item()}

# Serialize the prediction result into the desired response content type
def output_fn(predictions, response_content_type):
    return json.dumps(predictions)

def transform(args):
    data_filepath = os.path.join(args.values_dir, args.data_file)
    df = load_data(data_filepath)
    df["score"] = np.nan
    ds = FlowPhotoDataset(df, args.images_dir)

    model = model_fn(args.model_dir)
    model.eval()
    transform_image = model.module.transforms['eval']

    print(f"computing predictions (n={len(ds)})")
    with torch.no_grad():
        for idx, image in tqdm(enumerate(ds), total=len(ds)):
            pred = predict_fn(image[0], model)
            df.at[idx, "score"] = pred["score"]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--output-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--images-dir", type=str, default=os.environ["SM_CHANNEL_IMAGES"])
    parser.add_argument("--values-dir", type=str, default=os.environ["SM_CHANNEL_VALUES"])
    parser.add_argument(
        "--data-file",
        type=str,
        default="images.csv",
        help="filename of images CSV file",
    )

    args = parser.parse_args()
    results = transform(args)
    output_file = os.path.join(args.output_dir, "data", "predictions.csv")
    print(f"saving predictions: {output_file}")
    results.to_csv(os.path.join(output_file))

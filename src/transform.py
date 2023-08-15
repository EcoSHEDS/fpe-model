import os
import json
import torch
import io
from PIL import Image
from torchvision.transforms import ToTensor
from modules import ResNetRankNet

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
            input_shape=(3, checkpoint["params"]["input_shape"][0], checkpoint["params"]["input_shape"][1]),
            resnet_size=18,
            truncate=2,
            pretrained=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.params = params
        print("model loaded from checkpoint")

    model.eval()
    model.to(device)

def load_from_bytearray(request_body):
    image_as_bytes = io.BytesIO(request_body)
    image = Image.open(image_as_bytes)
    image_tensor = ToTensor()(image).unsqueeze(0)
    return image_tensor


def input_fn(request_body, request_content_type):
    # if set content_type as "image/jpg" or "application/x-npy",
    # the input is also a python bytearray
    print("input_fn(): params")
    print(params)
    if request_content_type == "image/jpg":
        image_tensor = load_from_bytearray(request_body)
    else:
        print("not support this type yet")
        raise ValueError("not support this type yet")
    return image_tensor


# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    print("predict_fn(): model.params")
    print(model.params)
    output = model.module.forward(input_object)
    pred = output.detach().cpu().numpy()

    return {"score": pred.item()}


# Serialize the prediction result into the desired response content type
def output_fn(predictions, response_content_type):
    return json.dumps(predictions)

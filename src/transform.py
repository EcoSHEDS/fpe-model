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
            input_shape=(3, params["input_shape"][0], params["input_shape"][1]),
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
    image_as_bytes = io.BytesIO(request_body)
    image = Image.open(image_as_bytes)
    image_tensor = ToTensor()(image).unsqueeze(0)
    return image_tensor


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
    output = model.module.forward_single(input_object)
    pred = output.detach().cpu().numpy()

    return {"score": pred.item()}


# Serialize the prediction result into the desired response content type
def output_fn(predictions, response_content_type):
    return json.dumps(predictions)

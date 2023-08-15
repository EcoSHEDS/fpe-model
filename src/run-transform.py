# Dev code
import os
import json
import torch
import io
from PIL import Image
from torchvision.transforms import ToTensor
from modules import ResNetRankNet

model_dir="data/WESTB0/model/model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

model_file = os.path.join(model_dir, "model.pth")
print(f"loading checkpoint: {model_file}")
with open(model_file, "rb") as f:
    checkpoint = torch.load(f, map_location=device)
    params = checkpoint["params"]
    print("params")
    print(params)
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

import argparse
import json
import os
import io
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor

from modules import ResNetRankNet
from datasets import FlowPhotoDataset
from utils import load_images_from_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image_from_bytearray(image_bytes: bytes) -> torch.Tensor:
    """Load an image from bytes and convert to tensor.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Image as tensor
    """
    image = Image.open(io.BytesIO(image_bytes))
    return ToTensor()(image)

def model_fn(model_dir: str) -> nn.Module:
    """Load model from checkpoint.
    
    Args:
        model_dir: Directory containing model checkpoint
        
    Returns:
        Loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model on device: {device}")

    model_path = Path(model_dir) / "model.pth"
    logger.info(f"Loading checkpoint: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    params = checkpoint["params"]
    
    model = ResNetRankNet(
        input_shape=(3, params["input_shape"][0], params["input_shape"][1]),
        transforms=checkpoint["transforms"],
        resnet_size=18,
        truncate=2,
    )
    model = nn.DataParallel(model, device_ids=[])
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model loaded successfully")

    return model.to(device).eval()

def input_fn(request_body: bytes, request_content_type: str) -> torch.Tensor:
    """Transform raw input into model input tensor.
    
    Args:
        request_body: Raw input bytes
        request_content_type: MIME type of input
        
    Returns:
        Input tensor for model
        
    Raises:
        ValueError: If content type is not supported
    """
    if request_content_type == "image/jpg":
        return load_image_from_bytearray(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_tensor: torch.Tensor, model: nn.Module) -> Dict[str, float]:
    """Generate prediction from model input.
    
    Args:
        input_tensor: Model input tensor
        model: Loaded model
        
    Returns:
        Dictionary containing prediction score
    """
    with torch.no_grad():
        transformed = model.module.transforms['eval'](input_tensor)
        output = model.module.forward_single(transformed.unsqueeze(0))
        score = output.detach().cpu().numpy().item()
        
    return {"score": score}

def output_fn(predictions: Dict[str, Any], response_content_type: str) -> str:
    """Transform prediction into API response.
    
    Args:
        predictions: Model predictions
        response_content_type: Desired response MIME type
        
    Returns:
        JSON string of predictions
    """
    return json.dumps(predictions)

def transform(args: argparse.Namespace) -> pd.DataFrame:
    """Run batch transform on dataset.
    
    Args:
        args: Command line arguments
        
    Returns:
        DataFrame with predictions
    """
    data_path = Path(args.values_dir) / args.data_file
    df = load_images_from_csv(data_path)
    df["score"] = np.nan
    ds = FlowPhotoDataset(df, args.images_dir)

    model = model_fn(args.model_dir)

    logger.info(f"Computing predictions for {len(ds)} images")
    with torch.no_grad():
        for idx, (image, _) in tqdm(enumerate(ds), total=len(ds)):
            pred = predict_fn(image, model)
            df.at[idx, "score"] = pred["score"]

    return df

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "./model"),
        help="Directory containing model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DIR", "./output"),
        help="Directory to save predictions"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_IMAGES", "./images"),
        help="Directory containing images"
    )
    parser.add_argument(
        "--values-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_VALUES", "./values"),
        help="Directory containing input CSV"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="images.csv",
        help="Filename of images CSV file"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run transform
    results = transform(args)
    
    # Save predictions
    output_path = output_dir / "predictions.csv"
    logger.info(f"Saving predictions to: {output_path}")
    results.to_csv(output_path)

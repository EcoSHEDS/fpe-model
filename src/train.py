import argparse
import ast
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import yaml

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Resize, RandomCrop, CenterCrop, RandomHorizontalFlip,
    RandomRotation, ColorJitter, Normalize, Compose, Grayscale, RandomPerspective, RandomAutocontrast, RandomEqualize, Lambda, GaussianBlur
)
from torchvision.transforms import functional

from datasets import FlowPhotoRankingPairsDataset
from losses import RankNetLoss
from modules import ResNetRankNet
from utils import set_seeds, load_pairs_from_csv, fit, validate

import mlflow

import platform

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ToUint8(nn.Module):
    """Convert float tensor to uint8."""
    def forward(self, x):
        return (x * 255).to(torch.uint8)

class ToFloat(nn.Module):
    """Convert uint8 tensor back to float."""
    def forward(self, x):
        return x.to(torch.float) / 255.0

class RandomGamma(nn.Module):
    """Apply random gamma adjustment.
    
    Args:
        gamma_min (float): Minimum gamma value
        gamma_max (float): Maximum gamma value
    """
    def __init__(self, gamma_min=0.8, gamma_max=1.2):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        
    def forward(self, x):
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(x, gamma)

def create_image_transforms(
    resize_shape: Tuple[int, int],
    input_shape: Tuple[int, int],
    transform_grayscale: bool = False,
    transform_random_crop: bool = True,
    transform_flip: bool = True,
    transform_rotate: bool = True,
    transform_rotate_degrees: int = 10,
    transform_color_jitter: bool = True,
    transform_color_jitter_brightness: float = 0.2,
    transform_color_jitter_contrast: float = 0.2,
    transform_color_jitter_saturation: float = 0.2,
    transform_color_jitter_hue: float = 0.1,
    transform_normalize: bool = True,
    channel_mean: Tuple[float, ...] = None,
    channel_stdev: Tuple[float, ...] = None,
    transform_perspective: bool = False,
    transform_perspective_distortion: float = 0.2,
    transform_auto_contrast: bool = False,
    transform_equalize: bool = False,
    transform_gamma: bool = False,
    transform_gamma_min: float = 0.8,
    transform_gamma_max: float = 1.2,
    transform_gaussian_blur: bool = False,
    transform_blur_kernel: int = 3,
) -> Dict[str, Compose]:
    """Create image transformation pipelines for training and evaluation.
    
    Args:
        resize_shape: Target size for initial resize
        input_shape: Final input shape after cropping
        transform_grayscale: Whether to convert to grayscale
        transform_random_crop: Whether to use random cropping
        transform_flip: Whether to use random horizontal flips
        transform_rotate: Whether to use random rotation
        transform_rotate_degrees: Max rotation degrees for augmentation
        transform_color_jitter: Whether to use color jitter
        transform_color_jitter_brightness: Max brightness adjustment (0-1)
        transform_color_jitter_contrast: Max contrast adjustment (0-1)
        transform_color_jitter_saturation: Max saturation adjustment (0-1)
        transform_color_jitter_hue: Max hue adjustment (0-0.5)
        transform_normalize: Whether to normalize pixel values
        channel_mean: Channel-wise means for normalization
        channel_stdev: Channel-wise standard deviations for normalization
        transform_perspective: Whether to use random perspective transforms
        transform_perspective_distortion: Max perspective distortion factor
        transform_auto_contrast: Whether to use random auto contrast
        transform_equalize: Whether to use random histogram equalization
        transform_gamma: Whether to use random gamma adjustment
        transform_gamma_min: Minimum gamma value for random adjustment
        transform_gamma_max: Maximum gamma value for random adjustment
        transform_gaussian_blur: Whether to use random gaussian blur
        transform_blur_kernel: Blur kernel size
    """
    image_transforms = {
        "train": [Resize(resize_shape)],
        "eval": [Resize(resize_shape)],
    }

    # 1. Basic spatial transforms first
    if transform_random_crop:
        image_transforms["train"].append(RandomCrop(input_shape))
    else:
        image_transforms["train"].append(CenterCrop(input_shape))
    
    if transform_flip:
        image_transforms["train"].append(RandomHorizontalFlip())
        
    if transform_rotate:
        image_transforms["train"].append(RandomRotation(transform_rotate_degrees))
      
    if transform_perspective:
        image_transforms["train"].append(
            RandomPerspective(distortion_scale=transform_perspective_distortion)
        )

    # 2. Color adjustments
    if transform_gamma:
        image_transforms["train"].append(
            RandomGamma(
                gamma_min=transform_gamma_min,
                gamma_max=transform_gamma_max
            )
        )

    if transform_auto_contrast:
        image_transforms["train"].append(RandomAutocontrast())
        
    if transform_equalize:
        image_transforms["train"].extend([
            ToUint8(),                 # Convert to uint8
            RandomEqualize(),          # Apply equalization
            ToFloat()                  # Convert back to float
        ])

    if transform_color_jitter:
        image_transforms["train"].append(ColorJitter(
            brightness=transform_color_jitter_brightness,
            contrast=transform_color_jitter_contrast,
            saturation=transform_color_jitter_saturation,
            hue=transform_color_jitter_hue
        ))

    # 3. Blur (after color adjustments, before grayscale)
    if transform_gaussian_blur:
        image_transforms["train"].append(
            GaussianBlur(kernel_size=transform_blur_kernel)
        )

    # 4. Grayscale conversion
    if transform_grayscale:
        image_transforms["train"].append(Grayscale(num_output_channels=3))
        image_transforms["eval"].append(Grayscale(num_output_channels=3))

    # 5. Normalization always last
    if transform_normalize and not transform_grayscale:
        image_transforms["train"].append(Normalize(channel_mean, channel_stdev))
        image_transforms["eval"].append(Normalize(channel_mean, channel_stdev))

    return {
        "train": Compose(image_transforms["train"]),
        "eval": Compose(image_transforms["eval"])
    }

def setup_model(
    input_shape: Tuple[int, int, int],
    transforms: Dict[str, Compose],
    device: torch.device,
    gpu_idx: int,
    resnet_size: int = 18,
    truncate: int = 2,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = False,
    weight_init: str = 'kaiming',
) -> nn.Module:
    """Initialize and configure the model.
    
    Args:
        input_shape: Model input dimensions (channels, height, width)
        transforms: Dictionary of image transforms
        device: Target device for model
        gpu_idx: GPU device index
        resnet_size: Size of ResNet backbone (18, 34, 50, 101, or 152)
        truncate: Number of ResNet layers to truncate
        dropout_rate: Dropout rate for fully connected layers
        use_batch_norm: Whether to use batch normalization
        weight_init: Weight initialization method ('none', 'kaiming', 'xavier', or 'normal')
    
    Returns:
        Configured model
    """
    model = ResNetRankNet(
        input_shape=input_shape,
        transforms=transforms,
        resnet_size=resnet_size,
        truncate=truncate,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    )

    # Initialize weights if specified
    logger.info("Initializing weights with method: %s", weight_init)
    if weight_init != "none":
        model.initialize_weights(method=weight_init)

    # Freeze resnet backbone initially
    for p in list(model.children())[0].parameters():
        p.requires_grad = False

    model = nn.DataParallel(model, device_ids=[gpu_idx])
    return model.to(device)

def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: SGD,
    train_loss: float,
    transforms: Dict[str, Compose],
    params: Dict[str, Any]
) -> None:
    """Save a model checkpoint.
    
    Args:
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer state to save
        train_loss: Current training loss
        transforms: Image transforms
        params: Additional parameters to save
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_loss": train_loss,
        "transforms": transforms,
        "params": params
    }, checkpoint_path)

def setup_mlflow(experiment_name: Optional[str] = None, run_name: Optional[str] = None, tracking_uri: Optional[str] = None) -> bool:
    """Setup MLflow tracking if available.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Name of the MLflow run
        tracking_uri: URI of the MLflow tracking server
        
    Returns:
        bool: Whether MLflow tracking is enabled
    """
    try:
        import mlflow
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info("MLflow tracking URI set to: %s", tracking_uri)
        
        if experiment_name and run_name:
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow experiment set to: %s", experiment_name)
            mlflow.start_run(run_name=run_name)
            logger.info("MLflow run set to: %s", run_name)
            return True
            
        return False
    except ImportError:
        logger.info("MLflow not available - training metrics will not be logged")
        return False

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_args_from_config(args: argparse.Namespace, config: dict, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Update arguments with values from config file.
    
    Args:
        args: Parsed command line arguments
        config: Configuration dictionary from YAML
        parser: ArgumentParser instance to check defaults
        
    Returns:
        Updated arguments namespace
    """
    args_dict = vars(args)
    
    for key, value in config.items():
        if key in args_dict and args_dict[key] == parser.get_default(key):
            args_dict[key] = value
            
    return argparse.Namespace(**args_dict)

def format_time(seconds: float) -> str:
    """Format time in seconds to hours:minutes:seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def set_deterministic_mode(seed: int = 1691) -> None:
    """Set up deterministic mode for reproducible results.
    
    Args:
        seed: Random seed to use
    """
    import random
    
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set PyTorch operations to be deterministic
    torch.use_deterministic_algorithms(True)
    
    # Set environment variable for deterministic operations
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    logger.info("Deterministic mode enabled with seed: %d", seed)

def train(args: argparse.Namespace) -> None:
    """Main training function."""
    logger.info("Starting training run")
    logger.info("Training arguments: %s", args.__dict__)

    # Enable deterministic mode if requested
    if args.deterministic:
        set_deterministic_mode(args.random_seed)
    
    # Setup MLflow if requested
    use_mlflow = setup_mlflow(
        experiment_name=args.mlflow_experiment_name,
        run_name=args.mlflow_run_name,
        tracking_uri=args.mlflow_tracking_uri
    )
    if use_mlflow:
        # Log parameters
        mlflow.log_params(args.__dict__)
        mlflow_params_path = Path(args.data_dir) / "mlflow-params.yaml"
        logger.info("MLflow params path: %s", mlflow_params_path)
        if mlflow_params_path.exists():
            logger.info("Loading additional MLflow params from %s", mlflow_params_path)
            with open(mlflow_params_path) as f:
                mlflow_params = yaml.safe_load(f)
                mlflow.log_params(mlflow_params)
        else:
            logger.info("No MLflow params file found at %s", mlflow_params_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(args.gpu))

    output_data_dir = Path(args.output_dir) / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_data_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # set_seeds(args.random_seed)

    # Load and split data
    logger.info("Loading dataset from %s", Path(args.data_dir) / args.data_file)
    pairs_df = load_pairs_from_csv(Path(args.data_dir) / args.data_file)
    train_df = pairs_df[pairs_df['split'] == "train"]
    val_df = pairs_df[pairs_df['split'] == "val"]
    logger.info("Dataset loaded - Train: %d samples, Validation: %d samples", 
                len(train_df), len(val_df))

    # Setup datasets and compute image stats
    logger.info("Computing image statistics from %d samples", args.transform_normalize_n)
    train_ds = FlowPhotoRankingPairsDataset(train_df, args.images_dir)
    img_mean, img_std = train_ds.compute_mean_std(args.transform_normalize_n)
    logger.info("Image statistics - Mean: %s, Std: %s", img_mean, img_std)

    # Calculate shapes using crop_ratio
    pair = train_ds.get_pair(0)
    image = train_ds.get_image(pair["filename_1"])
    aspect = image.shape[2] / image.shape[1]
    image_shape = image.shape
    resize_shape = [args.input_size, int(args.input_size * aspect)]
    input_shape = [
        int(args.input_size * args.crop_ratio), 
        int(args.input_size * args.crop_ratio * aspect)
    ]

    # Create transforms
    transforms = create_image_transforms(
        resize_shape=resize_shape,
        input_shape=input_shape,
        transform_grayscale=args.transform_grayscale,
        transform_random_crop=args.transform_random_crop,
        transform_flip=args.transform_flip,
        transform_rotate=args.transform_rotate,
        transform_rotate_degrees=args.transform_rotate_degrees,
        transform_color_jitter=args.transform_color_jitter,
        transform_color_jitter_brightness=args.transform_color_jitter_brightness,
        transform_color_jitter_contrast=args.transform_color_jitter_contrast,
        transform_color_jitter_saturation=args.transform_color_jitter_saturation,
        transform_color_jitter_hue=args.transform_color_jitter_hue,
        transform_normalize=args.transform_normalize,
        channel_mean=img_mean,
        channel_stdev=img_std,
        transform_perspective=args.transform_perspective,
        transform_perspective_distortion=args.transform_perspective_distortion,
        transform_auto_contrast=args.transform_auto_contrast,
        transform_equalize=args.transform_equalize,
        transform_gamma=args.transform_gamma,
        transform_gamma_min=args.transform_gamma_min,
        transform_gamma_max=args.transform_gamma_max,
        transform_gaussian_blur=args.transform_gaussian_blur,
        transform_blur_kernel=args.transform_blur_kernel,
    )
    
    train_ds.transform = transforms["train"]
    val_ds = FlowPhotoRankingPairsDataset(
        val_df,
        args.images_dir,
        transform=transforms["eval"]
    )

    # Create dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    eval_batch_size = args.eval_batch_size or args.batch_size
    val_dl = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Initialize model and training components
    model = setup_model(
        (3, input_shape[0], input_shape[1]),
        transforms,
        device,
        args.gpu,
        resnet_size=args.resnet_size,
        truncate=args.truncate,
        dropout_rate=args.dropout_rate,
        use_batch_norm=args.use_batch_norm,
        weight_init=args.weight_init
    )

    # Load from checkpoint if specified
    start_epoch = 0
    if args.from_checkpoint:
        if not os.path.exists(args.from_checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.from_checkpoint}")
        logger.info(f"Loading checkpoint from {args.from_checkpoint}")
        checkpoint = torch.load(args.from_checkpoint, map_location=device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize optimizer after model parameters are loaded
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get starting epoch
        logger.info(f"Resuming from previous checkpoint at epoch {checkpoint['epoch'] + 1}")
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Get model summary for logging
    model_summary = model.module.get_model_summary()
    logger.info("Model summary: %s", model_summary)

    criterion = RankNetLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, "min", 
        patience=args.scheduler_patience, 
        factor=args.scheduler_factor
    )

    # Training loop
    logger.info(f"Starting training from epoch {start_epoch + 1} for up to {args.epochs} epochs")
    metrics = {"epoch": [], "train_loss": [], "val_loss": []}
    min_val_loss = float('inf')
    best_epoch = None
    total_train_time = 0
    epoch_times = []
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        logger.info("Epoch %d/%d", epoch + 1, args.epochs)
        
        # Train
        start_time = time.time()
        train_loss = fit(model, criterion, optimizer, train_dl, device, epoch)
        train_time = time.time() - start_time
        logger.info("Training - Loss: %.4f (%.1f s)", train_loss, train_time)
        metrics["train_loss"].append(train_loss)
        metrics["epoch"].append(epoch)

        # Validate
        start_time = time.time()
        val_loss = validate(model, [criterion], val_dl, device)[0]
        val_time = time.time() - start_time
        logger.info("Validation - Loss: %.4f (%.1f s)", val_loss, val_time)
        metrics["val_loss"].append(val_loss)

        # Timing
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        total_train_time += epoch_time

        # Learning rate update
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            logger.info("Learning rate adjusted: %.6f -> %.6f", old_lr, new_lr)

        # Model saving
        checkpoint_path = Path(args.checkpoint_dir) / f"epoch_{epoch:02d}.pth"
        save_checkpoint(
            checkpoint_path, epoch, model, optimizer, train_loss,
            transforms,
            {
                "aspect": aspect,
                "input_shape": input_shape,
                "img_sample_mean": img_mean,
                "img_sample_std": img_std,
                "args": args.__dict__
            }
        )

        # Check for improvement
        if val_loss < (min_val_loss - args.early_stopping_min_delta):
            improvement = min_val_loss - val_loss
            if epoch == 0:
                logger.info("Initial checkpoint - Saving")
            else:
                logger.info("Model validation loss improved by %.4f - Saving checkpoint", improvement)
            best_model_path = Path(args.model_dir) / "model.pth"
            shutil.copy(checkpoint_path, best_model_path)
            min_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info("No improvement in validation loss for %d epochs", patience_counter)
            
            if patience_counter >= args.early_stopping_patience:
                logger.info("Early stopping triggered - No improvement for %d epochs", 
                          args.early_stopping_patience)
                break

        # Save metrics
        pd.DataFrame(metrics).to_csv(output_data_dir / "metrics.csv", index=False)

        # MLflow logging
        if use_mlflow:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": new_lr,
                "epoch_time": epoch_time,
            }, step=epoch)

        # Backbone unfreezing
        if (epoch + 1) == args.unfreeze_after:
            logger.info("Unfreezing CNN backbone at epoch %d", epoch + 1)
            for p in list(model.children())[0].parameters():
                p.requires_grad = True

        # Checkpoint management based on save frequency and keep limit
        if epoch % args.save_frequency == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"epoch_{epoch:02d}.pth"
            save_checkpoint(
                checkpoint_path, epoch, model, optimizer, train_loss,
                transforms,
                {
                    "aspect": aspect,
                    "input_shape": input_shape,
                    "img_sample_mean": img_mean,
                    "img_sample_std": img_std,
                    "args": args.__dict__
                }
            )

            # Remove old checkpoints if needed
            if args.keep_n_checkpoints > 0:
                checkpoints = sorted(Path(args.checkpoint_dir).glob("epoch_*.pth"))
                if len(checkpoints) > args.keep_n_checkpoints:
                    for checkpoint in checkpoints[:-args.keep_n_checkpoints]:
                        checkpoint.unlink()

    # Update completion message
    final_epoch = epoch + 1
    if patience_counter >= args.early_stopping_patience:
        logger.info("Training stopped early at epoch %d/%d", final_epoch, args.epochs)
    else:
        logger.info("Training completed all %d epochs", args.epochs)

    logger.info("Total training time: %s", format_time(total_train_time))
    logger.info("Average epoch time: %s", 
                format_time(sum(epoch_times) / len(epoch_times)))

    logger.info("Saving final outputs to %s", output_data_dir)
    
    if use_mlflow:
        mlflow.log_metrics({
            "total_train_time": total_train_time,
            "total_epochs": final_epoch,
            "train_n_pairs": len(train_df),
            "val_n_pairs": len(val_df),
            "best_epoch": best_epoch,
            "best_train_loss": float(f"{metrics['train_loss'][best_epoch]:.4f}"),
            "best_train_improvement": float(f"{metrics['train_loss'][0] - metrics['train_loss'][best_epoch]:.4f}"),
            "best_val_loss": float(f"{min_val_loss:.4f}"),
            "best_val_improvement": float(f"{metrics['val_loss'][0] - min_val_loss:.4f}")
        })

    # Save best metrics and training statistics
    output_dict = {
        'version': '1.0',  # Add versioning for output format
        
        'summary': {
            'initial_train_loss': float(f"{metrics['train_loss'][0]:.4f}"),
            'initial_val_loss': float(f"{metrics['val_loss'][0]:.4f}"),
            'best_epoch': best_epoch,
            'best_train_loss': float(f"{metrics['train_loss'][best_epoch]:.4f}"),
            'best_train_improvement': float(f"{metrics['train_loss'][0] - metrics['train_loss'][best_epoch]:.4f}"),
            'best_val_loss': float(f"{min_val_loss:.4f}"),
            'best_val_improvement': float(f"{metrics['val_loss'][0] - min_val_loss:.4f}"),
            'final_train_loss': float(f"{metrics['train_loss'][-1]:.4f}"),
            'final_val_loss': float(f"{metrics['val_loss'][-1]:.4f}"),
        },
        
        'timing': {
            'total_epochs': final_epoch,
            'total_seconds': total_train_time,
            'total_formatted': format_time(total_train_time),
            'per_epoch': {
                'average_seconds': float(f"{sum(epoch_times) / len(epoch_times):.2f}"),
                'average_formatted': format_time(sum(epoch_times) / len(epoch_times)),
                'all_epochs_seconds': [float(f"{t:.2f}") for t in epoch_times]
            }
        },
        
        'training': {
            'history': {
                'train_loss': [float(f"{loss:.4f}") for loss in metrics['train_loss']],
                'val_loss': [float(f"{loss:.4f}") for loss in metrics['val_loss']],
            },
            'learning_rate': {
                'initial': args.lr,
                'final': float(f"{optimizer.param_groups[0]['lr']:.6f}")
            },
            'hyperparameters': {
                'batch_size': args.batch_size,
                'optimizer': 'SGD',
                'momentum': args.momentum,
                'scheduler': 'ReduceLROnPlateau',
                'scheduler_patience': args.scheduler_patience,
                'scheduler_factor': args.scheduler_factor,
                'backbone_frozen_epochs': args.unfreeze_after
            }
        },
        
        'data': {
            'train': {
                'timestamps': {
                    'start': train_df['timestamp_1'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': train_df['timestamp_1'].max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'size': len(train_df),
                'batches': len(train_dl)
            },
            'val': {
                'timestamps': {
                    'start': val_df['timestamp_1'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': val_df['timestamp_1'].max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'size': len(val_df), 
                'batches': len(val_dl)
            },
            'image_preprocessing': {
                'channel_mean': [float(f"{m:.4f}") for m in img_mean],
                'channel_stdev': [float(f"{s:.4f}") for s in img_std],
                'image_shape': image_shape,
                'resize_shape': resize_shape,
                'input_shape': input_shape,
                'aspect_ratio': float(f"{aspect:.4f}"),
                'transforms': {
                    'normalize': args.transform_normalize,
                    'grayscale': args.transform_grayscale,
                    'random_crop': args.transform_random_crop,
                    'flip': args.transform_flip,
                    'rotate': {
                        'enabled': args.transform_rotate,
                        'degrees': args.transform_rotate_degrees
                    },
                    'color': {
                        'enabled': args.transform_color_jitter,
                        'brightness': args.transform_color_jitter_brightness,
                        'contrast': args.transform_color_jitter_contrast,
                        'saturation': args.transform_color_jitter_saturation,
                        'hue': args.transform_color_jitter_hue
                    },
                    'perspective': {
                        'enabled': args.transform_perspective,
                        'distortion': args.transform_perspective_distortion
                    },
                    'auto_contrast': {
                        'enabled': args.transform_auto_contrast,
                        'equalize': args.transform_equalize
                    },
                    'random_gamma': {
                        'enabled': args.transform_gamma,
                        'min': args.transform_gamma_min,
                        'max': args.transform_gamma_max
                    },
                    'gaussian_blur': {
                        'enabled': args.transform_gaussian_blur,
                        'blur_kernel': args.transform_blur_kernel
                    }
                },
            }
        },
        
        'model': {
            'architecture': 'ResNetRankNet',
            'base_model': f'resnet{args.resnet_size}',
            'parameters': model_summary,
            'hyperparameters': {
                'dropout_rate': args.dropout_rate,
                'batch_norm': args.use_batch_norm,
                'weight_init': args.weight_init,
                'truncate': args.truncate
            }
        },
        
        'environment': {
            'hardware': {
                'device': str(device),
                'gpu_index': args.gpu if torch.cuda.is_available() else None,
                'gpu_name': torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else None,
                'num_workers': args.num_workers
            },
            'software': {
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'python_version': platform.python_version()
            }
        },
        
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config_file': str(args.config) if args.config else None,
            'mlflow_experiment': args.mlflow_experiment_name if use_mlflow else None,
            'random_seed': args.random_seed
        },

        'args': vars(args)
    }
    with open(output_data_dir / "output.json", "w") as f:
        json.dump(output_dict, f, indent=2)

    logger.info("Training completed")
    
    # if use_mlflow:
    #     mlflow.log_artifacts(str(output_data_dir))

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # SageMaker parameters
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--checkpoint-dir", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--output-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--images-dir", type=str, default=os.environ["SM_CHANNEL_IMAGES"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    parser.add_argument("--num-workers", type=int, default=4, help="number of data loader workers")
    parser.add_argument("--gpu", type=int, default=0, help="index of the GPU to use")

    # hyperparameters
    parser.add_argument("--epochs", type=int, default=30, help="maximum number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--unfreeze-after", type=int, default=2,
        help="number of epochs after which to unfreeze model backbone",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="batch size of the train loader"
    )

    # transforms
    parser.add_argument(
        "--input-size", type=int, default=480,
        help="image input size to model",
    )
    parser.add_argument(
        "--crop-ratio", type=float, default=0.8,
        help="ratio of input size to use for cropping"
    )
    parser.add_argument(
        "--transform-grayscale",
        action="store_true",
        help="Convert images to grayscale before processing"
    )
    parser.add_argument(
        "--transform-normalize", type=bool, default=True,
        help="whether to normalize image inputs to model",
    )
    parser.add_argument(
        "--transform-normalize-n", type=int, default=1000,
        help="number of images to compute channel mean and stdev",
    )
    parser.add_argument(
        "--transform-random-crop", type=bool, default=True,
        help="whether to use random cropping during training",
    )
    parser.add_argument(
        "--transform-flip", type=bool, default=True,
        help="whether to use random horizontal flips during training",
    )
    parser.add_argument(
        "--transform-rotate", type=bool, default=True,
        help="whether to use random rotation during training",
    )
    parser.add_argument(
        "--transform-rotate-degrees", type=int, default=10,
        help="max rotation degrees for random rotation"
    )
    parser.add_argument(
        "--transform-color-jitter", type=bool, default=True,
        help="whether to use color jitter during training",
    )
    parser.add_argument(
        "--transform-color-jitter-brightness", type=float, default=0.2,
        help="max brightness adjustment for color jitter (0-1)",
    )
    parser.add_argument(
        "--transform-color-jitter-contrast", type=float, default=0.2,
        help="max contrast adjustment for color jitter (0-1)",
    )
    parser.add_argument(
        "--transform-color-jitter-saturation", type=float, default=0.2,
        help="max saturation adjustment for color jitter (0-1)",
    )
    parser.add_argument(
        "--transform-color-jitter-hue", type=float, default=0.1,
        help="max hue adjustment for color jitter (0-0.5)",
    )
    parser.add_argument(
        "--transform-perspective", type=bool, default=False,
        help="whether to use random perspective transforms"
    )
    parser.add_argument(
        "--transform-perspective-distortion", type=float, default=0.2,
        help="max perspective distortion factor"
    )
    parser.add_argument(
        "--transform-auto-contrast", type=bool, default=False,
        help="whether to use random auto contrast"
    )
    parser.add_argument(
        "--transform-equalize", type=bool, default=False,
        help="whether to use random histogram equalization"
    )
    parser.add_argument(
        "--transform-gamma", type=bool, default=False,
        help="whether to use random gamma adjustment"
    )
    parser.add_argument(
        "--transform-gamma-min", type=float, default=0.8,
        help="minimum gamma value for random adjustment"
    )
    parser.add_argument(
        "--transform-gamma-max", type=float, default=1.2,
        help="maximum gamma value for random adjustment"
    )
    parser.add_argument(
        "--transform-gaussian-blur", type=bool, default=False,
        help="whether to use random gaussian blur"
    )
    parser.add_argument(
        "--transform-blur-kernel", type=int, default=3,
        help="blur kernel size"
    )

    # input file
    parser.add_argument(
        "--data-file", type=str, default="train-pairs.csv",
        help="filename of CSV file with annotated image pairs",
    )

    # config file argument
    parser.add_argument(
        "--config", type=str, default=None,
        help="path to YAML configuration file"
    )

    # optimizer parameters
    parser.add_argument(
        "--momentum", type=float, default=0.9,
        help="momentum for SGD optimizer"
    )
    parser.add_argument(
        "--scheduler-patience", type=int, default=1,
        help="number of epochs to wait before reducing learning rate"
    )
    parser.add_argument(
        "--scheduler-factor", type=float, default=0.5,
        help="factor to reduce learning rate by"
    )

    # model architecture
    parser.add_argument(
        "--resnet-size", type=int, default=18,
        help="size of ResNet backbone (18, 34, 50, 101, or 152)"
    )
    parser.add_argument(
        "--truncate", type=int, default=2,
        help="number of ResNet layers to truncate"
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.0,
        help="dropout rate for fully connected layers"
    )
    parser.add_argument(
        "--use-batch-norm", action="store_true",
        help="use batch normalization in fully connected layers"
    )
    parser.add_argument(
        "--weight-init", type=str, default="none",
        choices=["none", "kaiming", "xavier", "normal"],
        help="weight initialization method for fully connected layers ('none' for no initialization)"
    )
        
    # validation parameters
    parser.add_argument(
        "--eval-batch-size", type=int, default=None,
        help="batch size for validation (defaults to training batch size)"
    )
    
    # Checkpoint parameters
    parser.add_argument(
        "--save-frequency", type=int, default=1,
        help="save checkpoint every N epochs"
    )
    parser.add_argument(
        "--keep-n-checkpoints", type=int, default=-1,
        help="number of checkpoints to keep (-1 for all)"
    )

    # Early stopping parameters
    parser.add_argument(
        "--early-stopping-patience", type=int, default=5,
        help="number of epochs to wait before early stopping"
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float, default=1e-4,
        help="minimum change to qualify as an improvement"
    )

    # MLflow parameters
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default="http://localhost:5000",
        help="MLflow tracking URI. If not provided, MLflow logging will be disabled."
    )
    parser.add_argument(
        "--mlflow-experiment-name", type=str, default=None,
        help="MLflow experiment name. If not provided, MLflow logging will be disabled."
    )
    parser.add_argument(
        "--mlflow-run-name", type=str, default=None,
        help="MLflow run name. If not provided, MLflow logging will be disabled."
    )

    # Reproducibility parameters
    parser.add_argument(
        "--deterministic", action="store_true",
        help="enable deterministic mode for reproducible results"
    )
    parser.add_argument(
        "--random-seed", type=int, default=1691,
        help="random seed for reproducibility"
    )

    # Add checkpoint argument
    parser.add_argument(
        "--from-checkpoint", type=Path, default=None,
        help="Path to checkpoint file to resume training from"
    )

    args = parser.parse_args()
    
    # Load and apply config file if specified
    if args.config is not None:
        logger.info("Loading config file: %s", args.config)
        config = load_config(args.config)
        args = update_args_from_config(args, config, parser)
    else:
        logger.warning("No config file provided - using default values")
        
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
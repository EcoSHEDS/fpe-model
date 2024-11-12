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
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Resize, RandomCrop, CenterCrop, RandomHorizontalFlip,
    RandomRotation, ColorJitter, Normalize, Compose, Grayscale
)

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

def create_image_transforms(
    resize_shape: Tuple[int, int],
    input_shape: Tuple[int, int],
    decolorize: bool = False,
    augmentation: bool = True,
    normalization: bool = True,
    means: Tuple[float, ...] = None,
    stds: Tuple[float, ...] = None,
    rotation_degrees: int = 10,
) -> Dict[str, Compose]:
    """Create image transformation pipelines for training and evaluation.
    
    Args:
        resize_shape: Target size for initial resize
        input_shape: Final input shape after cropping
        decolorize: Whether to convert to grayscale
        augmentation: Whether to apply data augmentation
        normalization: Whether to normalize pixel values
        means: Channel-wise means for normalization
        stds: Channel-wise standard deviations for normalization
        rotation_degrees: Max rotation degrees for augmentation
    
    Returns:
        Dictionary containing train and eval transform pipelines
    """
    image_transforms = {
        "train": [Resize(resize_shape)],
        "eval": [Resize(resize_shape)],
    }

    if decolorize:
        image_transforms["train"].append(Grayscale(num_output_channels=3))
        image_transforms["eval"].append(Grayscale(num_output_channels=3))

    if augmentation:
        image_transforms["train"].extend([
            RandomCrop(input_shape),
            RandomHorizontalFlip(),
            RandomRotation(rotation_degrees),
            ColorJitter(),
        ])
    else:
        image_transforms["train"].append(CenterCrop(input_shape))
    
    image_transforms["eval"].append(CenterCrop(input_shape))

    if normalization and not decolorize:
        image_transforms["train"].append(Normalize(means, stds))
        image_transforms["eval"].append(Normalize(means, stds))

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
) -> nn.Module:
    """Initialize and configure the model.
    
    Args:
        input_shape: Model input dimensions (channels, height, width)
        transforms: Dictionary of image transforms
        device: Target device for model
        gpu_idx: GPU device index
        resnet_size: Size of ResNet backbone (18, 34, 50, etc)
        truncate: Number of ResNet layers to truncate
    
    Returns:
        Configured model
    """
    model = ResNetRankNet(
        input_shape=input_shape,
        transforms=transforms,
        resnet_size=resnet_size,
        truncate=truncate
    )

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

def train(args: argparse.Namespace) -> None:
    """Main training function."""
    logger.info("Starting training run")
    logger.debug("Training arguments: %s", args.__dict__)
    
    # Setup MLflow if requested
    use_mlflow = setup_mlflow(
        experiment_name=args.mlflow_experiment_name,
        run_name=args.mlflow_run_name,
        tracking_uri=args.mlflow_tracking_uri
    )
    if use_mlflow:
        # Log parameters
        mlflow.log_params(args.__dict__)
        
        # Log config file if used
        if args.config:
            mlflow.log_param("config_file", args.config)
            # mlflow.log_artifact(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(args.gpu))

    output_data_dir = Path(args.output_dir) / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_data_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    set_seeds(args.random_seed)

    # Load and split data
    logger.info("Loading dataset from %s", Path(args.data_dir) / args.data_file)
    pairs_df = load_pairs_from_csv(Path(args.data_dir) / args.data_file)
    train_df = pairs_df[pairs_df['split'] == "train"]
    val_df = pairs_df[pairs_df['split'] == "val"]
    logger.info("Dataset loaded - Train: %d samples, Validation: %d samples", 
                len(train_df), len(val_df))

    # Setup datasets and compute image stats
    logger.info("Computing image statistics from %d samples", args.num_image_stats)
    train_ds = FlowPhotoRankingPairsDataset(train_df, args.images_dir)
    img_mean, img_std = train_ds.compute_mean_std(args.num_image_stats)
    logger.debug("Image statistics - Mean: %s, Std: %s", img_mean, img_std)

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

    # Create transforms with rotation parameter
    transforms = create_image_transforms(
        resize_shape, input_shape,
        means=img_mean, stds=img_std,
        decolorize=args.decolorize,
        augmentation=args.augment,
        normalization=args.normalize,
        rotation_degrees=args.rotation_degrees
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
        truncate=args.truncate
    )
    
    criterion = RankNetLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = ReduceLROnPlateau(optimizer, "min", 
                                 patience=args.scheduler_patience, 
                                 factor=args.scheduler_factor)

    # Training loop
    logger.info("Starting training for up to %d epochs", args.epochs)
    metrics = {"epoch": [], "train_loss": [], "val_loss": []}
    min_val_loss = float('inf')
    best_epoch = None
    total_train_time = 0
    epoch_times = []
    patience_counter = 0
    
    for epoch in range(args.epochs):
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
                "img_sample_std": img_std
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
                    "img_sample_std": img_std
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
                'means': [float(f"{m:.4f}") for m in img_mean],
                'stds': [float(f"{s:.4f}") for s in img_std],
                'image_shape': image_shape,
                'resize_shape': resize_shape,
                'input_shape': input_shape,
                'aspect_ratio': float(f"{aspect:.4f}"),
                'augmentation_enabled': args.augment,
                'normalization_enabled': args.normalize,
                'decolorize_enabled': args.decolorize
            }
        },
        
        'model': {
            'architecture': 'ResNetRankNet',
            'base_model': 'resnet18',
            'parameters': {
                'total': sum(p.numel() for p in model.parameters()),
                'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)
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
        }
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
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--unfreeze-after", type=int, default=2,
        help="number of epochs after which to unfreeze model backbone",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="batch size of the train loader"
    )
    parser.add_argument(
        "--random-seed", type=int, default=1691,
        help="random seed"
    )

    # transforms
    parser.add_argument(
        "--num-image-stats", type=int, default=1000,
        help="number of images to compute mean/stdev",
    )
    parser.add_argument(
        "--input-size", type=int, default=480,
        help="image input size to model",
    )
    parser.add_argument(
        "--decolorize", type=bool, default=False,
        help="remove image color channels",
    )
    parser.add_argument(
        "--normalize", type=bool, default=True,
        help="whether to normalize image inputs to model",
    )
    parser.add_argument(
        "--augment", type=bool, default=True,
        help="whether to use image augmentation during training",
    )

    # input file
    parser.add_argument(
        "--data-file", type=str, default="train-pairs.csv",
        help="filename of CSV file with annotated image pairs",
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
    parser.add_argument(
        "--mlflow-artifacts-dir", type=str, default=None,
        help="Local directory to store MLflow artifacts before upload"
    )

    # Add config file argument
    parser.add_argument(
        "--config", type=str, default=None,
        help="path to YAML configuration file"
    )

    # optimizer parameters
    parser.add_argument("--momentum", type=float, default=0.9,
                       help="momentum for SGD optimizer")
    parser.add_argument("--scheduler-patience", type=int, default=1,
                       help="number of epochs to wait before reducing learning rate")
    parser.add_argument("--scheduler-factor", type=float, default=0.5,
                       help="factor to reduce learning rate by")

    # Model parameters
    parser.add_argument("--resnet-size", type=int, default=18,
                       help="size of ResNet backbone (18, 34, 50, etc)")
    parser.add_argument("--truncate", type=int, default=2,
                       help="number of ResNet layers to truncate")
    
    # Transform parameters
    parser.add_argument("--crop-ratio", type=float, default=0.8,
                       help="ratio of input size to use for cropping")
    parser.add_argument("--rotation-degrees", type=int, default=10,
                       help="max rotation degrees for augmentation")
    
    # Validation parameters
    parser.add_argument("--eval-batch-size", type=int, default=None,
                       help="batch size for validation (defaults to training batch size)")
    
    # Checkpoint parameters
    parser.add_argument("--save-frequency", type=int, default=1,
                       help="save checkpoint every N epochs")
    parser.add_argument("--keep-n-checkpoints", type=int, default=-1,
                       help="number of checkpoints to keep (-1 for all)")

    # Early stopping parameters
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                       help="number of epochs to wait before early stopping")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4,
                       help="minimum change to qualify as an improvement")

    args = parser.parse_args()
    
    # Load and apply config file if specified
    if args.config is not None:
        config = load_config(args.config)
        args = update_args_from_config(args, config, parser)
        
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
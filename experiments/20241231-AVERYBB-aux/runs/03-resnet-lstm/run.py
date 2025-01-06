import argparse
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
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import (
    Resize, RandomCrop, CenterCrop, RandomHorizontalFlip,
    RandomRotation, ColorJitter, Normalize, Compose
)
import scipy.stats as stats
from torchvision.models import (
    resnet18, ResNet18_Weights
)
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MetricLogger(object):
    """Computes and tracks the average and current value of a metric.

    Attributes:
        val: current value
        sum: sum of all logged values
        count: number of logged values
        avg: average of all logged values
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count

class ResNet(nn.Module):
    """PyTorch ResNet architecture wrapper.
    
    A wrapper around torchvision's ResNet that allows truncating layers from the end
    of the network. Uses pretrained weights from torchvision.
    
    Attributes:
        model (nn.Module): The underlying ResNet model
    """

    def __init__(self, truncate=0):
        """Initialize the ResNet model.
        
        Args:
            truncate (int): Number of layers to remove from the end of the network.
                          Default is 0 (use full network).
        """
        super(ResNet, self).__init__()
        
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        if truncate > 0:
            self.model = nn.Sequential(*list(self.model.children())[:-truncate])

        self.model.eval()

    def forward(self, x):
        return self.model(x)

    def to_device(self, device):
        self.to(device)
        self.device = device
        return self

class ResNetRegression(nn.Module):
    """ResNet-based regression network for flow prediction with auxiliary data.
    
    A neural network that uses a truncated ResNet backbone followed by fully connected
    layers to perform regression. Processes auxiliary numerical features alongside 
    image data by concatenating them with the image features before the final layers.
    
    Attributes:
        input_shape (tuple): Expected input shape (channels, height, width)
        n_auxiliary (int): Number of auxiliary features
        transforms (list): List of transforms to apply to inputs
        resnetbody (nn.Module): Truncated ResNet backbone
        avgpool (nn.Module): Adaptive average pooling layer
        fclayers (nn.Module): Fully connected layers for regression
    """

    def __init__(
        self,
        input_shape=(3, 384, 512),
        n_auxiliary=2,  # Number of auxiliary features
        lstm_hidden_size=256,
        lstm_num_layers=2,
        lstm_dropout=0.1,
        transforms=[],
        truncate=2,
        num_hlayers=[256, 64],
    ):
        """Initialize the regression network.
        
        Args:
            input_shape (tuple): Expected input shape (channels, height, width)
            n_auxiliary (int): Number of auxiliary scalar features
            transforms (list): List of transforms to apply to inputs
            truncate (int): Number of layers to remove from ResNet
            num_hlayers (list): List of hidden layer sizes for the fully connected layers
        """
        super(ResNetRegression, self).__init__()
        self.input_shape = input_shape
        self.input_nchannels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.n_auxiliary = n_auxiliary
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.transforms = transforms

        # Initialize ResNet backbone
        self.resnet = ResNet(truncate=truncate)

        # Initialize LSTM
        self.lstm = nn.LSTM(
            input_size=n_auxiliary,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            batch_first=True
        )

        # Get number of features from ResNet
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Build fully connected layers
        # Note: 512 is from ResNet feature dim, add n_auxiliary for auxiliary features
        self.fclayer_modules = []
        for i, (in_features, out_features) in enumerate(zip([512 + lstm_hidden_size] + num_hlayers[:-1], num_hlayers)):
            self.fclayer_modules.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU()
            ])
        self.fclayer_modules.extend([nn.Linear(num_hlayers[-1], 1)])

        self.fclayers = nn.Sequential(*self.fclayer_modules)

    def forward(self, x, aux):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width)
            aux (torch.Tensor): Auxiliary features tensor of shape (batch_size, n_auxiliary)
            
        Returns:
            torch.Tensor: Predicted scalar value for each input in the batch
        """
        # Process image through ResNet
        resnet_features = self.resnet(x)
        resnet_features = self.avgpool(resnet_features)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)

        # Process auxiliary features through LSTM
        aux_features, _ = self.lstm(aux)
        aux_features = aux_features[:, -1, :]
        
        # Concatenate auxiliary features
        all_features = torch.cat([resnet_features, aux_features], dim=1)
        
        x = self.fclayers(all_features)
        return x.squeeze()

    def get_features(self, x, aux):
        """Extract features before the final layer.
        
        Args:
            x (torch.Tensor): Input image tensor
            aux (torch.Tensor): Auxiliary features tensor
            
        Returns:
            torch.Tensor: Features from the penultimate layer
        """
        resnet_features = self.resnet(x)
        resnet_features = self.avgpool(resnet_features)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)

        # Process auxiliary features through LSTM
        aux_features, _ = self.lstm(aux)
        aux_features = aux_features[:, -1, :]
        
        # Concatenate auxiliary features
        all_features = torch.cat([resnet_features, aux_features], dim=1)
        
        # Run through all layers except the last
        for layer in self.fclayer_modules[:-1]:
            all_features = layer(all_features)
        return all_features

class FlowPhotoDataset(Dataset):
    """Dataset class for flow photos with auxiliary data.
    
    Args:
        table (pd.DataFrame): images table with auxiliary columns
        images_dir (str): directory containing images
        aux_columns (list): names of columns containing auxiliary data
        transform (Compose): transforms to apply to images
    """

    def __init__(
        self,
        images_table,
        images_dir,
        aux_table,
        aux_columns=None,  # List of column names for auxiliary data
        sequence_length=5,
        transform=None,
    ) -> None:
        self.images_table = images_table
        self.images_dir = images_dir
        self.aux_table = aux_table
        self.aux_columns = aux_columns or []  # Default to empty list if None
        self.sequence_length = sequence_length
        self.transform = transform

        # Ensure dates are datetime objects
        self.images_table.loc[:, 'date'] = pd.to_datetime(self.images_table['date'])
        self.aux_table['date'] = pd.to_datetime(self.aux_table['date'])
        
        # Sort auxiliary table by date
        self.aux_table = self.aux_table.sort_values('date')
        
        # Verify we have sufficient auxiliary data
        self._validate_data()

        # Check for missing dates in auxiliary data
        date_range = pd.date_range(
            start=self.aux_table['date'].min(),
            end=self.aux_table['date'].max(),
            freq='D'
        )
        missing_dates = date_range.difference(self.aux_table['date'])
        if len(missing_dates) > 0:
            logger.warning(
                f"Found {len(missing_dates)} missing dates in auxiliary data. "
                "Values will be forward/backward filled."
            )

    def _validate_data(self):
        """Verify that auxiliary data covers the label period."""
        images_min_date = self.images_table['date'].min()
        images_max_date = self.images_table['date'].max()
        aux_min_date = self.aux_table['date'].min()
        aux_max_date = self.aux_table['date'].max()
        
        if images_min_date < aux_min_date:
            raise ValueError(f"Images start ({images_min_date}) before auxiliary data ({aux_min_date})")
        if images_max_date > aux_max_date:
            raise ValueError(f"Images end ({images_max_date}) after auxiliary data ({aux_max_date})")

    def _get_aux_sequence(self, date):
        """Get sequence of auxiliary data leading up to (and including) the given date."""
        # Find the exact date or the most recent date before it
        mask = self.aux_table['date'] <= date
        if not mask.any():
            raise ValueError(f"No auxiliary data found before {date}")
        
        # Get the most recent date index
        current_idx = mask.sum() - 1
        
        # Get the previous sequence_length days of data
        sequence_dates = pd.date_range(
            end=self.aux_table.iloc[current_idx]['date'],
            periods=self.sequence_length,
            freq='D'
        )
        
        # Get the data for these dates, using forward fill for missing dates
        sequence = pd.DataFrame(index=sequence_dates)
        sequence = sequence.join(
            self.aux_table.set_index('date')[self.aux_columns]
        ).ffill().bfill()
        
        return sequence.values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.images_table)

    def get_image(self, index):
        filename = self.images_table.iloc[index]["filename"]
        image_path = os.path.join(self.images_dir, filename)
        try:
            image = read_image(image_path)
            image = image / 255.0  # convert to float in [0,1]
            return image
        except:
            logger.error(f"Could not read image index {index} ({image_path})")

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, float]:
        image = self.get_image(index)
        if self.transform:
            image = self.transform(image)

        row = self.images_table.iloc[index]
        label = row["value"].astype(np.float32)
        aux = self._get_aux_sequence(row['date'])
        return image, aux, label

    def compute_mean_std(self, n=1000):
        """Compute RGB channel means and stds for image samples in the dataset."""
        means = np.zeros((3))
        stds = np.zeros((3))
        sample_size = min(len(self.images_table), n)
        sample_indices = np.random.choice(
            len(self.images_table), size=sample_size, replace=False
        )
        for idx in tqdm(sample_indices):
            image = self.get_image(idx)
            means += np.array(image.mean(dim=[1, 2]))
            stds += np.array(image.std(dim=[1, 2]))
        means = means / sample_size
        stds = stds / sample_size
        return means, stds

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

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_time(seconds: float) -> str:
    """Format time in seconds to hours:minutes:seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def fit(model, criterion, optimizer, train_dl, device, epoch_num=None, verbose=False):
    """Train model for one epoch."""
    model.train()
    batch_loss_logger = MetricLogger()
    batch_time_logger = MetricLogger()

    for bidx, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
        batch_starttime = time.time()
        inputs, aux, labels = batch  # Now unpacking 3 items
        if next(model.parameters()).is_cuda:
            inputs = inputs.to(device)
            aux = aux.to(device)
            labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, aux)  # Pass both inputs and aux
        loss = criterion(outputs, labels)
        batch_loss_logger.update(loss.item())
        loss.backward()
        optimizer.step()
        batch_endtime = time.time()
        batch_time_logger.update(batch_endtime - batch_starttime)

    return batch_loss_logger.avg

def validate(model, criterions, dl, device):
    """Calculate multiple criterion for a model on a dataset."""
    model.eval()
    criterion_loggers = [MetricLogger() for i in range(len(criterions))]
    
    with torch.no_grad():
        for bidx, batch in tqdm(enumerate(dl), total=len(dl)):
            model_outputs = {}
            for i, c in enumerate(criterions):
                if "outputs" not in model_outputs.keys():
                    inputs, aux, labels = batch  # Now unpacking 3 items
                    if next(model.parameters()).is_cuda:
                        inputs = inputs.to(device)
                        aux = aux.to(device)
                        labels = labels.to(device)
                    outputs = model(inputs, aux)  # Pass both inputs and aux
                    model_outputs["outputs"] = outputs
                else:
                    outputs = model_outputs["outputs"]
                cval = c(outputs, labels)
                criterion_loggers[i].update(cval.item())
                
    return [cl.avg for cl in criterion_loggers]

def train(args: argparse.Namespace) -> None:
    """Main training function."""
    logger.info("Starting training run")
    logger.info("Training arguments: %s", args.__dict__)

    # create directories
    Path(args.model_dir).mkdir(parents=False, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=False, exist_ok=True)
    Path(args.output_dir).mkdir(parents=False, exist_ok=True)

    # Enable deterministic mode if requested
    if args.seed:
        set_seed(args.seed)
        logger.info("Deterministic mode enabled with seed: %d", args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(args.gpu))

    output_data_dir = Path(args.output_dir) / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_data_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load and split data
    logger.info("Loading dataset from %s", Path(args.data_dir) / args.data_file)

    images_df = pd.read_csv(Path(args.data_dir) / args.data_file)

    logger.info("Loading auxiliary data from %s", Path(args.data_dir) / args.aux_file)
    aux_df = pd.read_csv(Path(args.data_dir) / args.aux_file)

    # Optionally shuffle filenames
    if args.shuffle:
        logger.info("Shuffling filenames")
        images_df['filename'] = images_df['filename'].sample(frac=1, random_state=args.seed, replace=False).values
        
    train_df = images_df[images_df['split'] == "train"]
    val_df = images_df[images_df['split'] == "val"]
    logger.info("Dataset loaded - Train: %d samples, Validation: %d samples", 
                len(train_df), len(val_df))

    # Add this after loading the DataFrame
    if args.aux_columns:
        missing_cols = [col for col in args.aux_columns if col not in aux_df.columns]
        if missing_cols:
            raise ValueError(f"Auxiliary columns not found in data: {missing_cols}")
        logger.info("Using auxiliary columns: %s", args.aux_columns)

    # Setup datasets and compute image stats
    logger.info("Computing image statistics from %d samples", args.transform_normalize_n)
    train_ds = FlowPhotoDataset(
        train_df,
        args.images_dir,
        aux_df,
        aux_columns=args.aux_columns,
        sequence_length=args.sequence_length
    )
    img_mean, img_std = train_ds.compute_mean_std(args.transform_normalize_n)
    logger.info("Image statistics - Mean: %s, Std: %s", img_mean, img_std)

    # Calculate shapes using crop_ratio
    image = train_ds.get_image(0)
    aspect = image.shape[2] / image.shape[1]
    image_shape = image.shape
    resize_shape = [args.input_size, int(args.input_size * aspect)]
    input_shape = [
        int(args.input_size * args.crop_ratio), 
        int(args.input_size * args.crop_ratio * aspect)
    ]

    # Create transforms
    transforms = {
        "train": Compose([
            Resize(resize_shape),
            RandomCrop(input_shape),
            RandomHorizontalFlip(),
            RandomRotation(degrees=10),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            Normalize(img_mean, img_std)
        ]),
        "eval": Compose([
            Resize(resize_shape),
            CenterCrop(input_shape),
            Normalize(img_mean, img_std)
        ])
    }
    
    train_ds.transform = transforms["train"]
    val_ds = FlowPhotoDataset(
        val_df,
        args.images_dir,
        aux_table=aux_df,
        aux_columns=args.aux_columns,
        sequence_length=args.sequence_length,
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

    model = ResNetRegression(
        input_shape=(3, input_shape[0], input_shape[1]),
        n_auxiliary=len(args.aux_columns),
        transforms=transforms["train"],
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        lstm_dropout=args.lstm_dropout
    )

    # Freeze resnet backbone initially
    for p in list(model.children())[0].parameters():
        p.requires_grad = False

    model = nn.DataParallel(model, device_ids=[args.gpu])
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", 
        patience=args.scheduler_patience, 
        factor=args.scheduler_factor
    )

    # Training loop
    start_epoch = 0
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
            best_model_path = Path(args.model_dir) / f"{args.mlflow_run}-model.pth"
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
        pd.DataFrame(metrics).to_csv(output_data_dir / f"{args.mlflow_run}-metrics.csv", index=False)

        # log metrics to mlflow
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

    # Update completion message
    final_epoch = epoch + 1
    if patience_counter >= args.early_stopping_patience:
        logger.info("Training stopped early at epoch %d/%d", final_epoch, args.epochs)
    else:
        logger.info("Training completed all %d epochs", args.epochs)

    logger.info("Total training time: %s", format_time(total_train_time))
    logger.info("Average epoch time: %s", 
                format_time(sum(epoch_times) / len(epoch_times)))

    mlflow.log_metrics({
        "total_train_time": total_train_time,
        "total_epochs": final_epoch,
        "train_n": len(train_df),
        "val_n": len(val_df),
        "best_epoch": best_epoch,
        "best_train_loss": float(f"{metrics['train_loss'][best_epoch]:.4f}"),
        # "best_train_improvement": float(f"{metrics['train_loss'][0] - metrics['train_loss'][best_epoch]:.4f}"),
        "best_val_loss": float(f"{min_val_loss:.4f}"),
        # "best_val_improvement": float(f"{metrics['val_loss'][0] - min_val_loss:.4f}")
    })

    logger.info("Training completed")

    # test model
    logger.info("Testing model")
    test_df = images_df.copy()
    test_ds = FlowPhotoDataset(
        test_df,
        args.images_dir,
        aux_df,
        aux_columns=args.aux_columns,
        sequence_length=args.sequence_length
    )
    model.eval()

    predictions = []
    with torch.no_grad():
        for image, aux, _ in tqdm(test_ds, desc="Generating predictions"):
            # Move image and aux to device and get prediction
            image = image.to(device)
            aux = torch.tensor(aux, dtype=torch.float32).to(device)  # Convert numpy array to tensor
            transformed = transforms['eval'](image)
            output = model.module.forward(transformed.unsqueeze(0), aux.unsqueeze(0))
            prediction = output.detach().cpu().numpy().item()
            predictions.append(prediction)
            
    test_df['prediction'] = np.array(predictions)
    test_df.to_csv(output_data_dir / f"{args.mlflow_run}-predictions.csv", index=False)

    metrics = []
    
    global_metrics = {}
    global_metrics["split"] = "global"
    global_metrics["tau"] = stats.kendalltau(predictions, test_df['value'])[0]
    global_metrics["rho"] = stats.spearmanr(predictions, test_df['value'])[0]
    global_metrics["mae"] = np.mean(np.abs(predictions - test_df['value']))
    global_metrics["rmse"] = np.sqrt(np.mean((predictions - test_df['value']) ** 2))
    metrics.append(global_metrics)
    print(global_metrics)

    mlflow.log_metrics({
        "test_global_tau": global_metrics["tau"],
        "test_global_rho": global_metrics["rho"],
        "test_global_mae": global_metrics["mae"],
        "test_global_rmse": global_metrics["rmse"],
    })

    # compute tau, rho, mae, rmse for each split in df
    for split in test_df['split'].unique():
        split_df = test_df[test_df['split'] == split]
        split_metrics = {}
        split_metrics["split"] = split
        split_metrics["tau"] = stats.kendalltau(split_df['prediction'], split_df['value'])[0]
        split_metrics["rho"] = stats.spearmanr(split_df['prediction'], split_df['value'])[0]
        split_metrics["mae"] = np.mean(np.abs(split_df['prediction'] - split_df['value']))
        split_metrics["rmse"] = np.sqrt(np.mean((split_df['prediction'] - split_df['value']) ** 2))
        metrics.append(split_metrics)

        x = {}
        x[f"test_split_{split}_tau"] = split_metrics["tau"]
        x[f"test_split_{split}_rho"] = split_metrics["rho"]
        x[f"test_split_{split}_mae"] = split_metrics["mae"]
        x[f"test_split_{split}_rmse"] = split_metrics["rmse"]
        mlflow.log_metrics(x)

    pd.DataFrame(metrics).to_csv(output_data_dir / f"{args.mlflow_run}-test-metrics.csv", index=False)

    for metric in metrics:
        logger.info("split: %s, tau: %.4f, rho: %.4f, mae: %.4f, rmse: %.4f", 
                    metric["split"], metric["tau"], metric["rho"], metric["mae"], metric["rmse"])

    logger.info("Testing completed")

    return model
    
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-dir", type=str, default='./model')
    parser.add_argument("--checkpoint-dir", type=str, default='./checkpoints')
    parser.add_argument("--output-dir", type=str, default='./output')
    parser.add_argument("--data-dir", type=str, default='./data')
    parser.add_argument("--images-dir", type=str, default='/home/jeff/data/fpe/images')
    parser.add_argument("--num-gpus", type=int, default=1)

    parser.add_argument("--num-workers", type=int, default=4, help="number of data loader workers")
    parser.add_argument("--gpu", type=int, default=0, help="index of the GPU to use")

    # hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="maximum number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
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
        "--transform-normalize-n", type=int, default=1000,
        help="number of images to compute channel mean and stdev",
    )

    # input file
    parser.add_argument(
        "--data-file", type=str, default="images.csv",
        help="filename of CSV file with images",
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
        "--truncate", type=int, default=2,
        help="number of ResNet layers to truncate"
    )

    # validation parameters
    parser.add_argument(
        "--eval-batch-size", type=int, default=None,
        help="batch size for validation (defaults to training batch size)"
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

    # Reproducibility parameters
    parser.add_argument(
        "--seed", type=int, default=1691,
        help="random seed for reproducibility"
    )

    # auxiliary columns
    parser.add_argument(
        "--aux-file", type=str, default="aux.csv",
        help="filename of CSV file with auxiliary data"
    )
    parser.add_argument(
        "--aux-columns", nargs="+", default=[],
        help="list of auxiliary columns to include"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=90,
        help="number of historical auxiliary timepoints to use"
    )

    # lstm parameters
    parser.add_argument(
        "--lstm-hidden-size", type=int, default=256,
        help="hidden size of LSTM"
    )
    parser.add_argument(
        "--lstm-num-layers", type=int, default=2,
        help="number of layers in LSTM"
    )
    parser.add_argument(
        "--lstm-dropout", type=float, default=0.1,
        help="dropout rate for LSTM"
    )

    # data shuffling
    parser.add_argument(
        "--shuffle", action="store_true",
        help="if set, randomly shuffle the image filenames"
    )

    # mlflow
    parser.add_argument(
        "--mlflow-run", type=str, required=True,
        help="name of the mlflow run"
    )
    parser.add_argument(
        "--mlflow-experiment", type=str, required=True,
        help="name of the mlflow experiment"
    )

    args = parser.parse_args()
    
    return args

def test(args: argparse.Namespace, model: nn.Module) -> None:
    pass

if __name__ == "__main__":
    args = parse_args()
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.start_run(run_name=args.mlflow_run)
    mlflow.log_params(args.__dict__)
    train(args)

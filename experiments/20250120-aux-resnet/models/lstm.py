import argparse
import json
import logging
import os
import time
import shutil
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import yaml

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
import scipy.stats as stats
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MetricLogger(object):
    """Computes and tracks the average and current value of a metric."""

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

class AuxiliaryRegression(nn.Module):
    """Neural network for regression using LSTM layers.
    
    Attributes:
        n_auxiliary (int): Number of auxiliary features
        lstm (nn.LSTM): LSTM layers
        fc (nn.Linear): Final fully connected layer
    """

    def __init__(
        self,
        n_auxiliary=2,  # Number of auxiliary features
        hidden_size=256,
        num_layers=2,
        dropout=0.1
    ):
        """Initialize the LSTM network.
        
        Args:
            n_auxiliary (int): Number of auxiliary scalar features
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout rate between LSTM layers
        """
        super(AuxiliaryRegression, self).__init__()
        self.n_auxiliary = n_auxiliary

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_auxiliary,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Final fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, aux):
        """Forward pass through the network.
        
        Args:
            aux (torch.Tensor): Auxiliary features tensor of shape (batch_size, sequence_length, n_auxiliary)
            
        Returns:
            torch.Tensor: Predicted scalar value for each input in the batch
        """
        # No need to reshape - aux should already be [batch_size, sequence_length, n_auxiliary]
        lstm_out, _ = self.lstm(aux)
        
        # Take the output from the last time step
        x = lstm_out[:, -1, :]
        
        # Final prediction
        x = self.fc(x)
        return x.squeeze()

    def get_features(self, aux):
        """Extract features before the final layer."""
        # Remove unsqueeze as input should already be 3D
        lstm_out, _ = self.lstm(aux)
        return lstm_out[:, -1, :]

class AuxiliaryDataset(Dataset):
    """Dataset class for auxiliary data with temporal lookback.
    
    Args:
        images_table (pd.DataFrame): Table with labeled values and timestamps
        aux_table (pd.DataFrame): Table with hourly auxiliary data
        aux_columns (list): Names of auxiliary columns to use
        sequence_length (int): Number of historical auxiliary timepoints to include
        timestamp_column (str): Name of the timestamp column in both tables
    """

    def __init__(
        self,
        images_table,
        aux_table,
        aux_columns=None,
        sequence_length=5,
        timestamp_column='timestamp'
    ) -> None:
        logger.info("Initializing AuxiliaryDataset")
        # Create explicit copies of the DataFrames
        self.images_table = images_table
        self.aux_table = aux_table
        self.aux_columns = aux_columns or []
        self.sequence_length = sequence_length
        self.timestamp_column = timestamp_column
        
        # Use .loc for assignments
        self.images_table[timestamp_column] = pd.to_datetime(self.images_table[timestamp_column], utc=True)
        self.aux_table[timestamp_column] = pd.to_datetime(self.aux_table[timestamp_column], utc=True)
        
        # Sort auxiliary table by timestamp
        self.aux_table = self.aux_table.sort_values(timestamp_column)
        
        # Verify we have sufficient auxiliary data
        self._validate_data()

        # Check for missing timestamps in auxiliary data
        time_range = pd.date_range(
            start=self.aux_table[timestamp_column].min(),
            end=self.aux_table[timestamp_column].max(),
            freq='h',  # Hourly frequency
            tz='UTC'
        )
        missing_times = time_range.difference(self.aux_table[timestamp_column])
        if len(missing_times) > 0:
            logger.warning(
                f"Found {len(missing_times)} missing timestamps in auxiliary data. "
                "Values will be forward/backward filled."
            )

    def _validate_data(self):
        """Verify that auxiliary data covers the image period."""
        logger.info("Validating auxiliary data")
        image_min_time = self.images_table[self.timestamp_column].min()
        image_max_time = self.images_table[self.timestamp_column].max()
        aux_min_time = self.aux_table[self.timestamp_column].min()
        aux_max_time = self.aux_table[self.timestamp_column].max()
        
        if image_min_time < aux_min_time:
            raise ValueError(f"Images start ({image_min_time}) before auxiliary data ({aux_min_time})")
        if image_max_time > aux_max_time:
            raise ValueError(f"Images end ({image_max_time}) after auxiliary data ({aux_max_time})")

    def _get_aux_sequence(self, timestamp):
        """Get sequence of auxiliary data leading up to (and including) the given timestamp."""
        # Find the exact timestamp or the most recent timestamp before it
        # logger.info("Getting auxiliary sequence for timestamp: %s", timestamp)
        mask = self.aux_table[self.timestamp_column] <= timestamp
        if not mask.any():
            raise ValueError(f"No auxiliary data found before {timestamp}")
        
        # Get the most recent timestamp index
        current_idx = mask.sum() - 1
        
        # Get the previous sequence_length hours of data
        sequence_times = pd.date_range(
            end=self.aux_table.iloc[current_idx][self.timestamp_column],
            periods=self.sequence_length,
            freq='h',  # Hourly frequency
            tz='UTC'
        )
        
        # Get the data for these timestamps, using forward fill for missing times
        sequence = pd.DataFrame(index=sequence_times)
        sequence = sequence.join(
            self.aux_table.set_index(self.timestamp_column)[self.aux_columns]
        ).ffill().bfill()
        
        return sequence.values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.images_table)

    def __getitem__(self, index) -> Tuple:
        row = self.images_table.iloc[index]
        timestamp = row[self.timestamp_column]
        label = row["value"].astype(np.float32)
        
        # Get historical auxiliary sequence
        aux_sequence = self._get_aux_sequence(timestamp)
        
        return aux_sequence, label

def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: Adam,
    train_loss: float,
    params: Dict[str, Any]
) -> None:
    """Save a model checkpoint.
    
    Args:
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer state to save
        train_loss: Current training loss
        params: Additional parameters to save
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_loss": train_loss,
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

def fit(model, criterion, optimizer, train_dl, device, epoch_num=None, scheduler=None):
    """Train model for one epoch."""
    model.train()
    batch_loss_logger = MetricLogger()
    
    for bidx, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
        aux, labels = batch
        if next(model.parameters()).is_cuda:
            aux = aux.to(device)
            labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(aux)
        loss = criterion(outputs, labels)
        batch_loss_logger.update(loss.item())
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # Update learning rate each batch
    
    return batch_loss_logger.avg

def validate(model, criterions, dl, device):
    """Calculate multiple criterion for a model on a dataset."""
    model.eval()
    criterion_loggers = [MetricLogger() for i in range(len(criterions))]
    
    with torch.no_grad():
        for bidx, batch in tqdm(enumerate(dl), total=len(dl)):
            aux, labels = batch
            if next(model.parameters()).is_cuda:
                aux = aux.to(device)
                labels = labels.to(device)
            outputs = model(aux)
            
            for i, c in enumerate(criterions):
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
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    output_data_dir = Path(args.output_dir) / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_data_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load data
    # Print files in data directory
    images_df = pd.read_csv(Path(args.data_dir) / args.images_file)
    logger.info(images_df.head())
    aux_df = pd.read_csv(Path(args.data_dir) / args.aux_file)
    
    # Handle aux_columns="*" by using all columns except timestamp
    if args.aux_columns == "*":
        args.aux_columns = [col for col in aux_df.columns if col != 'timestamp']
        logger.info(f"Using all auxiliary columns: {args.aux_columns}")
    elif isinstance(args.aux_columns, str):
        # If a single column was provided as string, convert to list
        args.aux_columns = [args.aux_columns]
    
    # Verify all requested columns exist
    missing_cols = [col for col in args.aux_columns if col not in aux_df.columns]
    if missing_cols:
        raise ValueError(f"Requested auxiliary columns not found in data: {missing_cols}")
    
    train_df = images_df[images_df['split'] == "train"].copy()
    val_df = images_df[images_df['split'] == "val"].copy()

    # Subsample if n_train or n_val is specified
    if args.n_train is not None:
        if args.n_train > len(train_df):
            logger.warning(
                f"Requested {args.n_train} training samples but only {len(train_df)} available. "
                "Using all available samples."
            )
        else:
            train_df = train_df.sample(n=args.n_train, random_state=args.seed)
            
    if args.n_val is not None:
        if args.n_val > len(val_df):
            logger.warning(
                f"Requested {args.n_val} validation samples but only {len(val_df)} available. "
                "Using all available samples."
            )
        else:
            val_df = val_df.sample(n=args.n_val, random_state=args.seed)

    logger.info("Dataset loaded - Train: %d samples, Validation: %d samples", 
                len(train_df), len(val_df))

    # Setup datasets with auxiliary lookback
    train_ds = AuxiliaryDataset(
        train_df,  # Now using subsampled train_df
        aux_df,
        aux_columns=args.aux_columns,
        sequence_length=args.sequence_length
    )
    val_ds = AuxiliaryDataset(
        val_df,   # Now using subsampled val_df
        aux_df,
        aux_columns=args.aux_columns,
        sequence_length=args.sequence_length
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

    model = AuxiliaryRegression(
        n_auxiliary=len(args.aux_columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    model = model.to(device)

    # Simple optimization setup
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # OneCycleLR scheduler - handles both warmup and annealing
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_dl),
        pct_start=0.3,  # Use 30% of training for warmup
    )
    # scheduler = None

    # Training loop
    start_epoch = 0
    logger.info(f"Starting training from epoch {start_epoch + 1} for up to {args.epochs} epochs")
    metrics = {"epoch": [], "train_loss": [], "val_loss": [], "learning_rate": []}
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
        train_loss = fit(model, criterion, optimizer, train_dl, device, epoch, scheduler)
        train_time = time.time() - start_time
        logger.info("Training - Loss: %.4f (%.1f s)", train_loss, train_time)
        metrics["train_loss"].append(train_loss)
        metrics["epoch"].append(epoch)
        metrics["learning_rate"].append(optimizer.param_groups[0]['lr'])

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

        # Model saving
        checkpoint_path = Path(args.checkpoint_dir) / f"epoch_{epoch:02d}.pth"
        save_checkpoint(
            checkpoint_path, epoch, model, optimizer, train_loss,
            {
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

        # log metrics to mlflow
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time": epoch_time,
        }, step=epoch)

        checkpoint_path = Path(args.checkpoint_dir) / f"epoch_{epoch:02d}.pth"
        save_checkpoint(
            checkpoint_path, epoch, model, optimizer, train_loss,
            {
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
    test_ds = AuxiliaryDataset(images_df.copy(), aux_df, aux_columns=args.aux_columns, sequence_length=args.sequence_length)
    model.eval()

    predictions = []
    with torch.no_grad():
        for aux, _ in tqdm(test_ds, desc="Generating predictions"):
            # Move aux to device and get prediction
            aux = torch.tensor(aux, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(aux)
            prediction = output.detach().cpu().numpy().item()
            predictions.append(prediction)
            
    images_df['prediction'] = np.array(predictions)
    images_df.to_csv(output_data_dir / "predictions.csv", index=False)

    metrics = []
    
    global_metrics = {}
    global_metrics["split"] = "global"
    global_metrics["tau"] = stats.kendalltau(predictions, images_df['value'])[0]
    global_metrics["rho"] = stats.spearmanr(predictions, images_df['value'])[0]
    global_metrics["mae"] = np.mean(np.abs(predictions - images_df['value']))
    global_metrics["rmse"] = np.sqrt(np.mean((predictions - images_df['value']) ** 2))
    metrics.append(global_metrics)
    print(global_metrics)

    mlflow.log_metrics({
        "test_global_tau": global_metrics["tau"],
        "test_global_rho": global_metrics["rho"],
        "test_global_mae": global_metrics["mae"],
        "test_global_rmse": global_metrics["rmse"],
    })

    # compute tau, rho, mae, rmse for each split in df
    for split in images_df['split'].unique():
        split_df = images_df[images_df['split'] == split]
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

    pd.DataFrame(metrics).to_csv(output_data_dir / "test-metrics.csv", index=False)

    for metric in metrics:
        logger.info("split: %s, tau: %.4f, rho: %.4f, mae: %.4f, rmse: %.4f", 
                    metric["split"], metric["tau"], metric["rho"], metric["mae"], metric["rmse"])

    logger.info("Testing completed")

    return model
    
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Basic arguments
    parser.add_argument("--model-dir", type=str, default='/opt/ml/model')
    parser.add_argument("--checkpoint-dir", type=str, default='/opt/ml/checkpoints')
    parser.add_argument("--output-dir", type=str, default='/opt/ml/output')
    parser.add_argument("--data-dir", type=str, default='/opt/ml/input/data/data')
    parser.add_argument("--images-dir", type=str, default='/opt/ml/input/data/images')
    parser.add_argument("--num-workers", type=int, default=4)

    # Add config file argument with default in data directory
    parser.add_argument("--config", type=str, default='config.yml',
                       help="Path to YAML config file, relative to data-dir")
    # hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)  # Default Adam learning rate
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=None)

    # input file
    parser.add_argument("--images-file", type=str, default="images.csv")

    # Early stopping parameters
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    # Reproducibility parameters
    parser.add_argument("--seed", type=int, default=1691)

    # auxiliary columns
    parser.add_argument("--aux-columns", type=str, nargs="+", default="*",
                       help="Names of auxiliary columns to use. Use '*' to include all columns except timestamp.")

    # mlflow
    parser.add_argument("--mlflow-run", type=str, required=True)
    parser.add_argument("--mlflow-experiment", type=str, required=True)

    # Add LSTM-specific parameters
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Add new data arguments
    parser.add_argument("--aux-file", type=str, default="aux.csv",
                      help="File containing daily auxiliary data")
    parser.add_argument("--sequence-length", type=int, default=24 * 7,
                      help="Number of historical auxiliary timepoints to use")
    
    # Sample size parameters
    parser.add_argument("--n-train", type=int, default=None,
                       help="Number of training samples to use. If None, use all available samples.")
    parser.add_argument("--n-val", type=int, default=None,
                       help="Number of validation samples to use. If None, use all available samples.")

    # First get the command line arguments
    args = parser.parse_args()
    
    # Construct full config path
    config_path = os.path.join(args.data_dir, args.config)
    
    # If config file exists, load it and update arguments
    if os.path.exists(config_path):
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Convert config to dict for easier manipulation
        args_dict = vars(args)
        
        # Update arguments from config, preserving command line arguments
        for key, value in config.items():
            # Convert dashes to underscores in config keys
            key = key.replace('-', '_')
            
            # Only update if not set in command line (None or default value)
            if key in args_dict:
                if args_dict[key] == parser.get_default(key):
                    args_dict[key] = value
            else:
                logger.warning(f"Unknown configuration key in config file: {key}")
        
        # Convert back to Namespace
        args = argparse.Namespace(**args_dict)
    else:
        logger.info(f"No config file found at {config_path}, using defaults and command line arguments")
    
    return args

def test(args: argparse.Namespace, model: nn.Module) -> None:
    pass

if __name__ == "__main__":
    args = parse_args()
    
    # Log the final configuration
    logger.info("Final configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.start_run(run_name=args.mlflow_run)
    mlflow.log_params(args.__dict__)
    train(args)

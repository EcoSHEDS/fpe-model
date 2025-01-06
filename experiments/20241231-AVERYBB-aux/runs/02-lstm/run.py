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
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        labels_table (pd.DataFrame): Table with labeled values and timestamps
        aux_table (pd.DataFrame): Table with daily auxiliary data
        aux_columns (list): Names of auxiliary columns to use
        sequence_length (int): Number of historical auxiliary timepoints to include
        date_column (str): Name of the date column in both tables
    """

    def __init__(
        self,
        labels_table,
        aux_table,
        aux_columns=None,
        sequence_length=5,
        date_column='date'
    ) -> None:
        self.labels_table = labels_table
        self.aux_table = aux_table
        self.aux_columns = aux_columns or []
        self.sequence_length = sequence_length
        self.date_column = date_column
        
        # Ensure dates are datetime objects
        self.labels_table[date_column] = pd.to_datetime(self.labels_table[date_column])
        self.aux_table[date_column] = pd.to_datetime(self.aux_table[date_column])
        
        # Sort auxiliary table by date
        self.aux_table = self.aux_table.sort_values(date_column)
        
        # Verify we have sufficient auxiliary data
        self._validate_data()

        # Check for missing dates in auxiliary data
        date_range = pd.date_range(
            start=self.aux_table[date_column].min(),
            end=self.aux_table[date_column].max(),
            freq='D'
        )
        missing_dates = date_range.difference(self.aux_table[date_column])
        if len(missing_dates) > 0:
            logger.warning(
                f"Found {len(missing_dates)} missing dates in auxiliary data. "
                "Values will be forward/backward filled."
            )

    def _validate_data(self):
        """Verify that auxiliary data covers the label period."""
        label_min_date = self.labels_table[self.date_column].min()
        label_max_date = self.labels_table[self.date_column].max()
        aux_min_date = self.aux_table[self.date_column].min()
        aux_max_date = self.aux_table[self.date_column].max()
        
        if label_min_date < aux_min_date:
            raise ValueError(f"Labels start ({label_min_date}) before auxiliary data ({aux_min_date})")
        if label_max_date > aux_max_date:
            raise ValueError(f"Labels end ({label_max_date}) after auxiliary data ({aux_max_date})")

    def _get_aux_sequence(self, date):
        """Get sequence of auxiliary data leading up to (and including) the given date."""
        # Find the exact date or the most recent date before it
        mask = self.aux_table[self.date_column] <= date
        if not mask.any():
            raise ValueError(f"No auxiliary data found before {date}")
        
        # Get the most recent date index
        current_idx = mask.sum() - 1
        
        # Get the previous sequence_length days of data
        sequence_dates = pd.date_range(
            end=self.aux_table.iloc[current_idx][self.date_column],
            periods=self.sequence_length,
            freq='D'
        )
        
        # Get the data for these dates, using forward fill for missing dates
        sequence = pd.DataFrame(index=sequence_dates)
        sequence = sequence.join(
            self.aux_table.set_index(self.date_column)[self.aux_columns]
        ).ffill().bfill()
        
        return sequence.values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels_table)

    def __getitem__(self, index) -> Tuple:
        row = self.labels_table.iloc[index]
        date = row[self.date_column]
        label = row["value"].astype(np.float32)
        
        # Get historical auxiliary sequence
        aux_sequence = self._get_aux_sequence(date)
        
        return aux_sequence, label

def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: SGD,
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

def fit(model, criterion, optimizer, train_dl, device, epoch_num=None, verbose=False):
    """Train model for one epoch."""
    model.train()
    batch_loss_logger = MetricLogger()
    batch_time_logger = MetricLogger()

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
        logger.info("GPU: %s", torch.cuda.get_device_name(args.gpu))

    output_data_dir = Path(args.output_dir) / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_data_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load data
    labels_df = pd.read_csv(Path(args.data_dir) / args.data_file)
    aux_df = pd.read_csv(Path(args.data_dir) / args.aux_file)
    
    train_df = labels_df[labels_df['split'] == "train"]
    val_df = labels_df[labels_df['split'] == "val"]

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

    model = nn.DataParallel(model, device_ids=[args.gpu])
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
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
    test_ds = AuxiliaryDataset(labels_df.copy(), aux_df, aux_columns=args.aux_columns, sequence_length=args.sequence_length)
    model.eval()

    predictions = []
    with torch.no_grad():
        for aux, _ in tqdm(test_ds, desc="Generating predictions"):
            # Move aux to device and get prediction
            aux = torch.tensor(aux, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
            # print(aux)
            output = model.module.forward(aux)  # aux is already [1, sequence_length, n_auxiliary]
            prediction = output.detach().cpu().numpy().item()
            predictions.append(prediction)
            
    labels_df['prediction'] = np.array(predictions)
    labels_df.to_csv(output_data_dir / f"{args.mlflow_run}-predictions.csv", index=False)

    metrics = []
    
    global_metrics = {}
    global_metrics["split"] = "global"
    global_metrics["tau"] = stats.kendalltau(predictions, labels_df['value'])[0]
    global_metrics["rho"] = stats.spearmanr(predictions, labels_df['value'])[0]
    global_metrics["mae"] = np.mean(np.abs(predictions - labels_df['value']))
    global_metrics["rmse"] = np.sqrt(np.mean((predictions - labels_df['value']) ** 2))
    metrics.append(global_metrics)
    print(global_metrics)

    mlflow.log_metrics({
        "test_global_tau": global_metrics["tau"],
        "test_global_rho": global_metrics["rho"],
        "test_global_mae": global_metrics["mae"],
        "test_global_rmse": global_metrics["rmse"],
    })

    # compute tau, rho, mae, rmse for each split in df
    for split in labels_df['split'].unique():
        split_df = labels_df[labels_df['split'] == split]
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
    
    # Keep basic arguments
    parser.add_argument("--model-dir", type=str, default='./model')
    parser.add_argument("--checkpoint-dir", type=str, default='./checkpoints')
    parser.add_argument("--output-dir", type=str, default='./output')
    parser.add_argument("--data-dir", type=str, default='./data')
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)

    # hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=None)

    # input file
    parser.add_argument("--data-file", type=str, default="labels.csv")

    # optimizer parameters
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler-patience", type=int, default=1)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)

    # Early stopping parameters
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    # Reproducibility parameters
    parser.add_argument("--seed", type=int, default=1691)

    # auxiliary columns
    parser.add_argument("--aux-columns", nargs="+", required=True)

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
    parser.add_argument("--sequence-length", type=int, default=90,
                      help="Number of historical auxiliary timepoints to use")
    
    # Sample size parameters
    parser.add_argument("--n-train", type=int, default=None,
                       help="Number of training samples to use. If None, use all available samples.")
    parser.add_argument("--n-val", type=int, default=None,
                       help="Number of validation samples to use. If None, use all available samples.")

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

import argparse
import ast
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
from tqdm import tqdm
import yaml

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import scipy.stats as stats
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, outputs1, outputs2, targets):
        """Calculate RankNet loss.
        
        Args:
            outputs1: Scores for first items in pairs (batch_size,)
            outputs2: Scores for second items in pairs (batch_size,)
            targets: Target labels as -1, 0, or 1 (batch_size,)
            
        Returns:
            Loss value
        """
        # calculate difference between scores of each image pair
        diff = outputs1 - outputs2  # Shape: (batch_size,)
        
        # calculate probability that sample i should rank higher than sample j
        Pij = torch.sigmoid(diff) 

        # map target labels to probabilities
        target_probs = (targets + 1) / 2  # Map {-1, 0, 1} to {0, 0.5, 1}
        
        # Binary cross entropy between predicted and target probabilities
        return self.bceloss(Pij, target_probs)

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

class AuxDataset(Dataset):
    """Dataset for auxiliary timeseries data.
    
    Args:
        table (pd.DataFrame): Data table with values to predict
        aux_data (pd.DataFrame): Auxiliary timeseries data
        aux_model (str): Type of auxiliary data processing ('encoder' or 'lstm')
        aux_sequence_length (int): Number of previous days to use for LSTM sequences
        aux_timestep (str): Timestep for LSTM sequence: 'D' for daily or 'H' for hourly
    """

    def __init__(
        self,
        table,
        aux_data,
        aux_model,
        aux_sequence_length=30,
        aux_timestep="D",
    ) -> None:
        self.table = table
        self.aux_data = aux_data
        self.aux_model = aux_model
        self.aux_lstm_sequence_length = aux_sequence_length
        self.aux_timestep = aux_timestep
        self.aux_time_col = 'date' if self.aux_timestep == "D" else 'timestamp'
        logger.info(f"Aux time col: {self.aux_time_col}")

    def __len__(self) -> int:
        return len(self.table)

    def get_aux_sequence(self, timestamp):
        """Get sequence of auxiliary data leading up to given date.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            Array of auxiliary data values for sequence_length timesteps up to and including date
        """
        if self.aux_data is None:
            return None
            
        # Find the index of the target timestamp
        if self.aux_timestep == "H":
            timestamp = timestamp.floor('h')
        target_idx = self.aux_data[self.aux_data[self.aux_time_col] == timestamp].index
        
        if len(target_idx) == 0:
            # If timestamp not found, return zeros
            num_features = len(self.aux_data.columns) - 1
            return np.zeros((self.aux_lstm_sequence_length, num_features), dtype=np.float32)
            
        target_idx = target_idx[0]
        
        # Calculate start index for sequence
        start_idx = max(0, target_idx - self.aux_lstm_sequence_length + 1)
        
        # Get the sequence data directly
        sequence = self.aux_data.iloc[start_idx:target_idx + 1]
        
        # If we don't have enough history, pad with zeros
        if len(sequence) < self.aux_lstm_sequence_length:
            num_features = len(self.aux_data.columns) - 1
            padding = np.zeros((self.aux_lstm_sequence_length - len(sequence), num_features), dtype=np.float32)
            sequence_data = sequence.drop(self.aux_time_col, axis=1).values.astype(np.float32)
            return np.vstack([padding, sequence_data])
            
        return sequence.drop(self.aux_time_col, axis=1).values.astype(np.float32)

    def get_aux_data(self, timestamp):
        """Get auxiliary data for a given timestamp.
        
        Args:
            timestamp: Timestamp to get data for
            
        Returns:
            Array of auxiliary data values
        """
        if self.aux_data is None:
            return None
            
        if self.aux_model == "lstm":
            return self.get_aux_sequence(timestamp)
        else:
            if self.aux_timestep == "H":
                timestamp = timestamp.floor('h')
            aux_row = self.aux_data[self.aux_data[self.aux_time_col] == timestamp]
            if len(aux_row) == 0:
                logger.warning(f"No auxiliary data found for timestamp {timestamp}")
                return None
            return aux_row.drop(self.aux_time_col, axis=1).values.astype(np.float32)[0]

    def __getitem__(self, index) -> Tuple:
        label = self.table.iloc[index]["value"].astype(np.float32)
        timestamp = pd.to_datetime(self.table.iloc[index][self.aux_time_col])
        aux_features = self.get_aux_data(timestamp)
        
        if aux_features is not None:
            return aux_features, label
        else:
            raise ValueError(f"No auxiliary data found for timestamp {timestamp}")

class AuxRankingPairsDataset(Dataset):
    """Dataset for pairs of auxiliary data for ranking.
    
    Args:
        table (pd.DataFrame): Table with pairs of data to rank
        aux_data (pd.DataFrame): Auxiliary timeseries data
        aux_model (str): Type of auxiliary data processing ('encoder' or 'lstm')
        aux_sequence_length (int): Number of previous days to use for LSTM sequences
        aux_timestep (str): Timestep for LSTM sequence: 'D' for daily or 'H' for hourly
    """

    def __init__(
        self,
        table,
        aux_data,
        aux_model,
        aux_sequence_length=30,
        aux_timestep="D",
    ) -> None:
        self.table = table
        self.aux_data = aux_data
        self.aux_model = aux_model
        self.aux_lstm_sequence_length = aux_sequence_length
        self.aux_timestep = aux_timestep
        self.aux_time_col = 'date' if self.aux_timestep == "D" else 'timestamp'

    def get_aux_sequence(self, timestamp):
        """Get sequence of auxiliary data leading up to given date.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            Array of auxiliary data values for sequence_length timesteps up to and including date
        """
        if self.aux_data is None:
            return None
            
        # Find the index of the target timestamp
        target_idx = self.aux_data[self.aux_data[self.aux_time_col] == timestamp].index
        
        if len(target_idx) == 0:
            # If timestamp not found, return zeros
            num_features = len(self.aux_data.columns) - 1
            return np.zeros((self.aux_lstm_sequence_length, num_features), dtype=np.float32)
            
        target_idx = target_idx[0]
        
        # Calculate start index for sequence
        start_idx = max(0, target_idx - self.aux_lstm_sequence_length + 1)
        
        # Get the sequence data directly
        sequence = self.aux_data.iloc[start_idx:target_idx + 1]
        
        # If we don't have enough history, pad with zeros
        if len(sequence) < self.aux_lstm_sequence_length:
            num_features = len(self.aux_data.columns) - 1
            padding = np.zeros((self.aux_lstm_sequence_length - len(sequence), num_features), dtype=np.float32)
            sequence_data = sequence.drop(self.aux_time_col, axis=1).values.astype(np.float32)
            return np.vstack([padding, sequence_data])
            
        return sequence.drop(self.aux_time_col, axis=1).values.astype(np.float32)

    def get_aux_data(self, timestamp):
        """Get auxiliary data for a given timestamp.
        
        Args:
            timestamp: Timestamp to get data for
            
        Returns:
            Array of auxiliary data values
        """
        if self.aux_data is None:
            return None
            
        if self.aux_model == "lstm":
            return self.get_aux_sequence(timestamp)
        else:
            if self.aux_timestep == "H":
                timestamp = timestamp.floor('h')
            aux_row = self.aux_data[self.aux_data[self.aux_time_col] == timestamp]
            if len(aux_row) == 0:
                logger.warning(f"No auxiliary data found for timestamp {timestamp}")
                return None
            return aux_row.drop(self.aux_time_col, axis=1).values.astype(np.float32)[0]

    def get_pair(self, index):
        return self.table.iloc[index]

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        pair = self.get_pair(idx)
        label = pair['label']

        aux1 = self.get_aux_data(pair[self.aux_time_col + '_1'])
        aux2 = self.get_aux_data(pair[self.aux_time_col + '_2'])
        
        if aux1 is not None and aux2 is not None:
            return aux1, aux2, label
        else:
            raise ValueError(f"No auxiliary data found for timestamps {pair[self.aux_time_col + '_1']} and/or {pair[self.aux_time_col + '_2']}")

class AuxRegressionNet(nn.Module):
    """Regression network for auxiliary data processing.
    
    A neural network that processes auxiliary data using either an encoder or LSTM architecture
    followed by fully connected layers for regression.
    
    Attributes:
        aux_input_size (int): Size of auxiliary input features
        aux_model (str): Type of auxiliary data processing ('encoder' or 'lstm')
        aux_encoder (nn.Module): Neural network to process auxiliary features
        fclayers (nn.Module): Fully connected layers for regression
    """

    def __init__(
        self,
        aux_input_size,
        aux_model,
        aux_encoder_layers=[64, 32],
        aux_encoder_dropout=0.0,
        aux_lstm_hidden=64,
        aux_lstm_layers=2,
        aux_lstm_dropout=0.0,
        num_hlayers=[64, 32],
        dropout_rate=0.0,
    ):
        """Initialize the regression network.
        
        Args:
            aux_input_size (int): Size of auxiliary input features
            aux_model (str): Type of auxiliary data processing ('encoder' or 'lstm')
            aux_encoder_layers (list): List of hidden layer sizes for auxiliary encoder
            aux_encoder_dropout (float): Dropout rate for auxiliary encoder layers
            aux_lstm_hidden (int): Hidden size for LSTM auxiliary encoder
            aux_lstm_layers (int): Number of LSTM layers
            aux_lstm_dropout (float): Dropout rate for LSTM layers
            num_hlayers (list): List of hidden layer sizes for final layers
            dropout_rate (float): Dropout rate for fully connected layers
        """
        super(AuxRegressionNet, self).__init__()
        
        if aux_model not in ["encoder", "lstm"]:
            raise ValueError(f"Invalid auxiliary model: {aux_model}. Must be 'encoder' or 'lstm'")
            
        self.aux_input_size = aux_input_size
        self.aux_model = aux_model

        # Initialize auxiliary processing
        if aux_model == "encoder":
            logger.info(f"Using auxiliary encoder with input size {aux_input_size} and layers {aux_encoder_layers}")
            aux_layers = []
            prev_size = aux_input_size
            
            for h in aux_encoder_layers:
                aux_layers.extend([
                    nn.Linear(prev_size, h),
                    nn.ReLU(),
                ])
                if aux_encoder_dropout > 0:
                    aux_layers.append(nn.Dropout(aux_encoder_dropout))
                prev_size = h
                
            self.aux_encoder = nn.Sequential(*aux_layers)
            aux_output_size = aux_encoder_layers[-1]
        else:  # lstm
            logger.info(f"Using LSTM auxiliary encoder with input size {aux_input_size}, hidden size {aux_lstm_hidden}, layers {aux_lstm_layers}, and dropout {aux_lstm_dropout}")
            self.aux_encoder = nn.LSTM(
                input_size=aux_input_size,
                hidden_size=aux_lstm_hidden,
                num_layers=aux_lstm_layers,
                dropout=aux_lstm_dropout if aux_lstm_layers > 1 else 0,
                batch_first=True
            )
            aux_output_size = aux_lstm_hidden

        # Build fully connected layers with optional dropout
        self.fclayer_modules = []
        for i, (in_features, out_features) in enumerate(zip([aux_output_size] + num_hlayers[:-1], num_hlayers)):
            self.fclayer_modules.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU()
            ])
            if dropout_rate > 0:
                self.fclayer_modules.append(nn.Dropout(dropout_rate))
        self.fclayer_modules.extend([nn.Linear(num_hlayers[-1], 1)])

        self.fclayers = nn.Sequential(*self.fclayer_modules)

    def forward_single(self, aux):
        """Forward pass through the network.
        
        Args:
            aux (torch.Tensor): Auxiliary features tensor. For LSTM, shape should be
                (batch_size, sequence_length, features)
            
        Returns:
            torch.Tensor: Predicted scalar value for each input in the batch
        """
        if self.aux_model == "lstm":
            # Process sequence through LSTM
            aux_out, _ = self.aux_encoder(aux)
            # Take the last output
            x = aux_out[:, -1, :]
        else:  # encoder
            x = self.aux_encoder(aux)
            
        x = self.fclayers(x)
        return x.squeeze()

    def forward(self, aux):
        """Forward pass through the network.
        
        Args:
            aux (torch.Tensor): Auxiliary features tensor
            
        Returns:
            torch.Tensor: Predicted scalar value for each input in the batch
        """
        return self.forward_single(aux)

    def get_features(self, aux):
        """Extract features before the final layer.
        
        Args:
            aux (torch.Tensor): Auxiliary features tensor
            
        Returns:
            torch.Tensor: Features from the penultimate layer
        """
        if self.aux_model == "lstm":
            aux_out, _ = self.aux_encoder(aux)
            x = aux_out[:, -1, :]
        else:  # encoder
            x = self.aux_encoder(aux)
        
        # Run through all layers except the last
        for layer in self.fclayer_modules[:-1]:
            x = layer(x)
        return x

    def get_model_summary(self):
        """Get a summary of the model architecture and parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'fc_layers': [m.out_features for m in self.fclayer_modules if isinstance(m, nn.Linear)],
            'aux_input_size': self.aux_input_size,
            'aux_model': self.aux_model
        }

class AuxRankNet(AuxRegressionNet):
    """Network for learning to rank using auxiliary data.
    
    Extends AuxRegressionNet to handle pairs of auxiliary data for learning to rank.
    The network produces scalar scores that can be compared to determine relative ranking.
    
    Inherits all attributes from AuxRegressionNet.
    """

    def __init__(
        self,
        aux_input_size,
        aux_model,
        aux_encoder_layers=[64, 32],
        aux_encoder_dropout=0.0,
        aux_lstm_hidden=64,
        aux_lstm_layers=2,
        aux_lstm_dropout=0.0,
        num_hlayers=[64, 32],
        dropout_rate=0.0,
    ):
        """Initialize the ranking network.
        
        Args:
            aux_input_size (int): Size of auxiliary input features
            aux_model (str): Type of auxiliary data processing ('encoder' or 'lstm')
            aux_encoder_layers (list): List of hidden layer sizes for auxiliary encoder
            aux_encoder_dropout (float): Dropout rate for auxiliary encoder layers
            aux_lstm_hidden (int): Hidden size for LSTM auxiliary encoder
            aux_lstm_layers (int): Number of LSTM layers
            aux_lstm_dropout (float): Dropout rate for LSTM layers
            num_hlayers (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate for fully connected layers
        """
        super().__init__(
            aux_input_size=aux_input_size,
            aux_model=aux_model,
            aux_encoder_layers=aux_encoder_layers,
            aux_encoder_dropout=aux_encoder_dropout,
            aux_lstm_hidden=aux_lstm_hidden,
            aux_lstm_layers=aux_lstm_layers,
            aux_lstm_dropout=aux_lstm_dropout,
            num_hlayers=num_hlayers,
            dropout_rate=dropout_rate,
        )
        
    def forward(self, aux1, aux2):
        """Forward pass for a pair of inputs.
        
        Args:
            aux1 (torch.Tensor): First auxiliary input
            aux2 (torch.Tensor): Second auxiliary input
            
        Returns:
            tuple: Predicted scalar scores for both inputs
        """
        output1 = self.forward_single(aux1)
        output2 = self.forward_single(aux2)
        return output1, output2

def fit(model, criterion, optimizer, train_dl, device, epoch_num=None, verbose=False):
    """Train model for one epoch."""
    model.train()
    batch_loss_logger = MetricLogger()
    batch_time_logger = MetricLogger()

    for bidx, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
        batch_starttime = time.time()

        if isinstance(criterion, (torch.nn.MarginRankingLoss, RankNetLoss)):
            # RankNet training
            aux1, aux2, labels = batch
            if next(model.parameters()).is_cuda:
                aux1 = aux1.to(device)
                aux2 = aux2.to(device)
                labels = labels.to(device)
            optimizer.zero_grad()
            outputs1, outputs2 = model(aux1, aux2)
            loss = criterion(outputs1, outputs2, labels)
        else:
            # Regression training
            aux_features, labels = batch
            if next(model.parameters()).is_cuda:
                aux_features = aux_features.to(device)
                labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(aux_features)
            loss = criterion(outputs, labels)

        batch_loss_logger.update(loss.item())
        loss.backward()
        optimizer.step()

        batch_endtime = time.time()
        batch_time_logger.update(batch_endtime - batch_starttime)

        if verbose and (bidx % 10 == 9):
            print(
                f"[Epoch {epoch_num} Batch {bidx}]\t{batch_time_logger.sum:.2f} s\t{batch_loss_logger.avg:.4f}"
            )

    print(
        f"[Epoch {epoch_num}|train]\t{batch_time_logger.sum:.2f} s\t{batch_loss_logger.avg:.4f}"
    )

    return batch_loss_logger.avg

class PairwiseRankAccuracy(torch.nn.Module):
    def __init__(self):
        super(PairwiseRankAccuracy, self).__init__()

    def forward(self, outputs_i, outputs_j, targets, boundaries=[0.33, 0.66]):
        oij = outputs_i - outputs_j
        Pij = torch.sigmoid(oij)
        preds = torch.zeros_like(targets)
        preds = torch.where(Pij < boundaries[0], -1, preds)
        preds = torch.where(Pij > boundaries[1], 1, preds)
        total = targets.size(0)
        correct = torch.eq(preds, targets).sum()
        return 100 * correct / float(total)

def validate(model, criterions, dl, device):
    """Calculate multiple criterion for a model on a dataset."""
    model.eval()
    criterion_loggers = [MetricLogger() for i in range(len(criterions))]
    
    with torch.no_grad():
        for bidx, batch in tqdm(enumerate(dl), total=len(dl)):
            model_outputs = {}
            for i, c in enumerate(criterions):
                if isinstance(c, (torch.nn.MarginRankingLoss, RankNetLoss, PairwiseRankAccuracy)):
                    # RankNet validation
                    if "outputs1" not in model_outputs.keys():
                        aux1, aux2, labels = batch
                        if next(model.parameters()).is_cuda:
                            aux1 = aux1.to(device)
                            aux2 = aux2.to(device)
                            labels = labels.to(device)
                        outputs1, outputs2 = model(aux1, aux2)
                        model_outputs["outputs1"] = outputs1
                        model_outputs["outputs2"] = outputs2
                        model_outputs["labels"] = labels
                    else:
                        outputs1, outputs2 = model_outputs["outputs1"], model_outputs["outputs2"]
                        labels = model_outputs["labels"]
                    cval = c(outputs1, outputs2, labels)
                else:
                    # Regression validation
                    aux_features, labels = batch
                    if next(model.parameters()).is_cuda:
                        aux_features = aux_features.to(device)
                        labels = labels.to(device)
                    outputs = model(aux_features)
                    cval = c(outputs, labels)
                criterion_loggers[i].update(cval.item())
                
    return [cl.avg for cl in criterion_loggers]

def load_pairs_from_csv(pairs_file, timestep="D"):
    df = pd.read_csv(pairs_file)
    
    if timestep == "D":
        time_col = 'date'
    else:  # "H"
        time_col = 'timestamp'
    
    df[f'{time_col}_1'] = pd.to_datetime(df[f'{time_col}_1'], utc=True)
    df[f'{time_col}_2'] = pd.to_datetime(df[f'{time_col}_2'], utc=True)

    return df

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
        logger.info(f"Loaded config from {config_path}: {config}")
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

    # Parse aux_encoder_layers from string to list
    if hasattr(args, 'aux_encoder_layers'):
        args.aux_encoder_layers = ast.literal_eval(args.aux_encoder_layers)
    if hasattr(args, 'num_hlayers'):
        args.num_hlayers = ast.literal_eval(args.num_hlayers)

    # Enable deterministic mode if requested
    if args.random_seed:
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
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    output_data_dir = Path(args.output_dir) / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_data_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load auxiliary data
    logger.info("Loading auxiliary data from %s", Path(args.data_dir) / args.aux_file)
    aux_data = pd.read_csv(Path(args.data_dir) / args.aux_file)
    
    # Convert time column based on timestep
    if args.aux_timestep == "D":
        time_col = 'date'
        aux_data[time_col] = pd.to_datetime(aux_data[time_col], utc=True)
    else:  # "H"
        time_col = 'timestamp'
        aux_data[time_col] = pd.to_datetime(aux_data[time_col], utc=True)
        
    aux_input_size = len(aux_data.columns) - 1  # Subtract time column
    logger.info("Auxiliary data loaded - %d features, timestep: %s", 
               aux_input_size, args.aux_timestep)

    if args.model_type == "ranknet":
        # Load and split ranking pairs data
        logger.info("Loading ranking pairs dataset from %s", Path(args.data_dir) / args.pairs_file)
        pairs_df = load_pairs_from_csv(Path(args.data_dir) / args.pairs_file, args.aux_timestep)
        train_df = pairs_df[pairs_df['split'] == "train"]
        val_df = pairs_df[pairs_df['split'] == "val"]
        logger.info("Dataset loaded - Train: %d samples, Validation: %d samples", 
                    len(train_df), len(val_df))

        # Setup datasets for ranking
        train_ds = AuxRankingPairsDataset(
            train_df, 
            aux_data=aux_data,
            aux_model=args.aux_model,
            aux_sequence_length=args.aux_lstm_sequence_length,
            aux_timestep=args.aux_timestep,
        )
        val_ds = AuxRankingPairsDataset(
            val_df,
            aux_data=aux_data,
            aux_model=args.aux_model,
            aux_sequence_length=args.aux_lstm_sequence_length,
            aux_timestep=args.aux_timestep,
        )
        
        # Initialize RankNet model
        model = AuxRankNet(
            aux_input_size=aux_input_size,
            aux_model=args.aux_model,
            aux_encoder_layers=args.aux_encoder_layers,
            aux_encoder_dropout=args.aux_encoder_dropout,
            aux_lstm_hidden=args.aux_lstm_hidden,
            aux_lstm_layers=args.aux_lstm_layers,
            aux_lstm_dropout=args.aux_lstm_dropout,
            num_hlayers=args.num_hlayers,
            dropout_rate=args.dropout_rate,
        ).to(device)
        
        criterion = RankNetLoss()
        
    else:  # regression
        # Load and split regression data
        logger.info("Loading regression dataset from %s", Path(args.data_dir) / args.images_file)
        images_df = pd.read_csv(Path(args.data_dir) / args.images_file)
        if args.aux_timestep == "D":
            images_df['date'] = pd.to_datetime(images_df['date'], utc=True)
        else:  # "H"
            images_df['timestamp'] = pd.to_datetime(images_df['timestamp'], utc=True)
            
        train_df = images_df[images_df['split'] == "train"]
        val_df = images_df[images_df['split'] == "val"]
        logger.info("Dataset loaded - Train: %d samples, Validation: %d samples", 
                    len(train_df), len(val_df))

        # Setup datasets for regression
        train_ds = AuxDataset(
            train_df, 
            aux_data=aux_data,
            aux_model=args.aux_model,
            aux_sequence_length=args.aux_lstm_sequence_length,
            aux_timestep=args.aux_timestep,
        )
        val_ds = AuxDataset(
            val_df,
            aux_data=aux_data,
            aux_model=args.aux_model,
            aux_sequence_length=args.aux_lstm_sequence_length,
            aux_timestep=args.aux_timestep,
        )
        
        # Initialize regression model
        model = AuxRegressionNet(
            aux_input_size=aux_input_size,
            aux_model=args.aux_model,
            aux_encoder_layers=args.aux_encoder_layers,
            aux_encoder_dropout=args.aux_encoder_dropout,
            aux_lstm_hidden=args.aux_lstm_hidden,
            aux_lstm_layers=args.aux_lstm_layers,
            aux_lstm_dropout=args.aux_lstm_dropout,
            num_hlayers=args.num_hlayers,
            dropout_rate=args.dropout_rate,
        ).to(device)
        
        criterion = nn.MSELoss()

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

    # Get model summary for logging
    model_summary = model.get_model_summary()
    logger.info("Model summary: %s", model_summary)

    # Initialize optimizer and scheduler
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", 
        patience=args.scheduler_patience, 
        factor=args.scheduler_factor
    )

    # Training loop
    logger.info(f"Starting training for up to {args.epochs} epochs")
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

        # MLflow logging
        if use_mlflow:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": new_lr,
                "epoch_time": epoch_time,
            }, step=epoch)

        # Checkpoint management based on save frequency and keep limit
        if epoch % args.save_frequency == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"epoch_{epoch:02d}.pth"
            save_checkpoint(
                checkpoint_path, epoch, model, optimizer, train_loss,
                {
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

    logger.info("Training completed")

    # test model
    logger.info("Testing model")

    # Load best model for testing
    best_model_path = Path(args.model_dir) / "model.pth"
    logger.info("Loading best model from %s", best_model_path)
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load test data
    logger.info("Loading test data from %s", Path(args.data_dir) / args.images_file)
    images_df = pd.read_csv(Path(args.data_dir) / args.images_file)
    if args.aux_timestep == "D":
        images_df['date'] = pd.to_datetime(images_df['date'], utc=True)
    else:  # "H"
        images_df['timestamp'] = pd.to_datetime(images_df['timestamp'], utc=True)
    
    test_ds = AuxDataset(
        images_df, 
        aux_data=aux_data,
        aux_model=args.aux_model,
        aux_sequence_length=args.aux_lstm_sequence_length,
        aux_timestep=args.aux_timestep,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    model.eval()

    predictions = []
    with torch.no_grad():
        for aux_features, _ in tqdm(test_dl, desc="Generating predictions"):
            aux_features = aux_features.to(device)
            outputs = model.forward_single(aux_features)
            batch_predictions = outputs.detach().cpu().numpy()
            predictions.extend(batch_predictions)
            
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
    
    # Directory arguments
    parser.add_argument("--model-dir", type=str, default='/opt/ml/model')
    parser.add_argument("--checkpoint-dir", type=str, default='/opt/ml/checkpoints')
    parser.add_argument("--output-dir", type=str, default='/opt/ml/output')
    parser.add_argument("--data-dir", type=str, default='/opt/ml/input/data/data')

    parser.add_argument("--num-workers", type=int, default=4, help="number of data loader workers")

    # Model type
    parser.add_argument(
        "--model-type", type=str, choices=["ranknet", "regression"], default="ranknet",
        help="Type of model architecture to use: 'ranknet' for pairwise ranking or 'regression' for direct value prediction"
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="maximum number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="batch size of the train loader"
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=None,
        help="batch size for validation (defaults to training batch size)"
    )

    # Input files
    parser.add_argument(
        "--pairs-file", type=str, default="pairs.csv",
        help="filename of CSV file with annotated pairs",
    )
    parser.add_argument(
        "--images-file", type=str, default="images.csv",
        help="filename of CSV file with data table",
    )
    parser.add_argument(
        "--aux-file", type=str,
        help="filename of CSV file with auxiliary timeseries data",
    )

    # Config file argument
    parser.add_argument(
        "--config", type=str, default="config.yml",
        help="path to YAML configuration file (default: config.yml)"
    )

    # Optimizer parameters
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

    # Model architecture
    parser.add_argument(
        "--aux-model", type=str, choices=["encoder", "lstm"],
        help="Type of auxiliary data processing: 'encoder' for learned encoding or 'lstm' for sequential processing"
    )
    parser.add_argument(
        "--aux-encoder-layers", type=str, default="[64, 32]",
        help="List of hidden layer sizes for auxiliary encoder (when aux-model='encoder')"
    )
    parser.add_argument(
        "--aux-encoder-dropout", type=float, default=0.0,
        help="Dropout rate for auxiliary encoder layers"
    )
    parser.add_argument(
        "--aux-lstm-hidden", type=int, default=64,
        help="Hidden size for LSTM auxiliary encoder"
    )
    parser.add_argument(
        "--aux-lstm-layers", type=int, default=2,
        help="Number of LSTM layers"
    )
    parser.add_argument(
        "--aux-lstm-dropout", type=float, default=0.0,
        help="Dropout rate for LSTM layers"
    )
    parser.add_argument(
        "--aux-lstm-sequence-length", type=int, default=30,
        help="Number of previous timesteps of auxiliary data to use for LSTM"
    )
    parser.add_argument(
        "--aux-timestep", type=str, default="D", choices=["D", "H"],
        help="Timestep for LSTM sequence: 'D' for daily or 'H' for hourly"
    )
    
    # Fully connected layers
    parser.add_argument(
        "--num-hlayers", type=str, default="[64, 32]",
        help="List of hidden layer sizes for final layers"
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.0,
        help="Dropout rate for fully connected layers"
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
        "--early-stopping-min-delta", type=float, default=1e-3,
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
        "--random-seed", type=int, default=None,
        help="random seed for reproducibility"
    )

    args = parser.parse_args()
    
    # Load and apply config file if specified
    if args.config is not None:
        logger.info("Loading config file: %s", args.config)
        config = load_config(Path(args.data_dir) / args.config)
        args = update_args_from_config(args, config, parser)
    else:
        logger.warning("No config file provided - using default values")
        
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)

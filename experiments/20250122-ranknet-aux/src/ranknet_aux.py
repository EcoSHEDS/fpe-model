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
from torchvision.io import read_image
from torchvision.transforms import (
    Resize, RandomCrop, CenterCrop, RandomHorizontalFlip,
    RandomRotation, ColorJitter, Normalize, Compose, Grayscale, RandomPerspective, RandomAutocontrast, RandomEqualize, Lambda, GaussianBlur, functional
)
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights
)
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

class FlowPhotoDataset(Dataset):
    """
    Args:
        table (pd.DataFrame): images table
        images_dir (str): directory containing images
        transform (Compose): transforms to apply to images
        aux_data (pd.DataFrame, optional): auxiliary timeseries data
        aux_model (str): Type of auxiliary data processing ('concat', 'encoder', or 'lstm')
        aux_sequence_length (int): Number of previous days to use for LSTM sequences
        aux_timestep (str): Timestep for LSTM sequence: 'D' for daily or 'H' for hourly
    """

    def __init__(
        self,
        table,
        images_dir,
        transform=None,
        aux_data=None,
        aux_model=None,
        aux_sequence_length=30,
        aux_timestep="D",
    ) -> None:
        self.table = table
        self.images_dir = images_dir
        self.transform = transform
        self.aux_data = aux_data
        self.aux_model = aux_model
        self.aux_lstm_sequence_length = aux_sequence_length
        self.aux_timestep = aux_timestep
        self.aux_time_col = 'date' if self.aux_timestep == "D" else 'timestamp'
        logger.info(f"Aux time col: {self.aux_time_col}")

    def __len__(self) -> int:
        return len(self.table)

    def get_image(self, index):
        filename = self.table.iloc[index]["filename"]
        image_path = os.path.join(self.images_dir, filename)
        try:
            image = read_image(image_path)
            image = image / 255.0  # convert to float in [0,1]
            return image
        except:
            logger.error(f"Could not read image index {index} ({image_path})")
            raise

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
        if self.aux_data is None or self.aux_model is None:
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
        image = self.get_image(index)
        label = self.table.iloc[index]["value"].astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
            
        # Get auxiliary data if available
        if self.aux_data is not None:
            timestamp = pd.to_datetime(self.table.iloc[index][self.aux_time_col])
            aux_features = self.get_aux_data(timestamp)
            if aux_features is not None:
                return image, aux_features, label
                
        return image, label

    def compute_mean_std(self, n=1000):
        """Compute RGB channel means and stds for image samples in the dataset."""
        means = np.zeros((3))
        stds = np.zeros((3))
        sample_size = min(len(self.table), n)
        sample_indices = np.random.choice(
            len(self.table), size=sample_size, replace=False
        )
        for idx in tqdm(sample_indices):
            image = self.get_image(idx)
            means += np.array(image.mean(dim=[1, 2]))
            stds += np.array(image.std(dim=[1, 2]))
        means = means / sample_size
        stds = stds / sample_size
        return means, stds

class FlowPhotoRankingPairsDataset():
    def __init__(
        self,
        table,
        images_dir,
        aux_data=None,
        transform=None,
        aux_sequence_length=30,
        aux_model=None,
        aux_timestep="D",
    ) -> None:
        self.table = table
        self.images_dir = images_dir
        self.aux_data = aux_data
        self.transform = transform
        self.aux_lstm_sequence_length = aux_sequence_length
        self.aux_model = aux_model
        self.aux_timestep = aux_timestep
        self.aux_time_col = 'date' if self.aux_timestep == "D" else 'timestamp'

    def get_image(self, filename):
        image_path = os.path.join(self.images_dir, filename)

        try:
            image = read_image(image_path)
            image = image / 255.0  # convert to float in [0,1]
            return image
        except:
            print(f"Could not read image ({filename})")
            raise

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
        if self.aux_data is None or self.aux_model is None:
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

    def compute_mean_std(self, n=1000):
        """Compute RGB channel means and stds for image samples in the dataset."""
        means = np.zeros((3))
        stds = np.zeros((3))
        sample_size = min(len(self.table), n)
        sample_indices = np.random.choice(
            len(self.table), size=sample_size, replace=False
        )
        for idx in tqdm(sample_indices):
            pair = self.get_pair(idx)
            image = self.get_image(pair['filename_1'])
            means += np.array(image.mean(dim=[1, 2]))
            stds += np.array(image.std(dim=[1, 2]))
        means = means / sample_size
        stds = stds / sample_size
        return means, stds

    def get_pair(self, index):
        return self.table.iloc[index]

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        pair = self.get_pair(idx)
        image1 = self.get_image(pair['filename_1'])
        image2 = self.get_image(pair['filename_2'])
        label = pair['label']

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)        

        # Get auxiliary data if available
        aux1 = self.get_aux_data(pair[self.aux_time_col + '_1'])
        aux2 = self.get_aux_data(pair[self.aux_time_col + '_2'])
        
        if aux1 is not None and aux2 is not None:
            return image1, image2, aux1, aux2, label
        else:
            return image1, image2, label

class ResNet(nn.Module):
    """PyTorch ResNet architecture wrapper.
    
    A wrapper around torchvision's ResNet that allows truncating layers from the end
    of the network. Uses pretrained weights from torchvision.
    
    Attributes:
        model (nn.Module): The underlying ResNet model
    """

    def __init__(self, size=18, truncate=0):
        """Initialize the ResNet model.
        
        Args:
            size (int): Size of ResNet backbone (18, 34, 50, 101, or 152). Default is 18.
            truncate (int): Number of layers to remove from the end of the network.
                          Default is 0 (use full network).
        """
        super(ResNet, self).__init__()
        
        # Map size to model and weights
        resnet_models = {
            18: (resnet18, ResNet18_Weights.DEFAULT),
            34: (resnet34, ResNet34_Weights.DEFAULT),
            50: (resnet50, ResNet50_Weights.DEFAULT),
            101: (resnet101, ResNet101_Weights.DEFAULT),
            152: (resnet152, ResNet152_Weights.DEFAULT)
        }
        
        if size not in resnet_models:
            raise ValueError(f"ResNet size {size} not supported. Choose from: {list(resnet_models.keys())}")
            
        model_fn, weights = resnet_models[size]
        self.model = model_fn(weights=weights)
        
        if truncate > 0:
            self.model = nn.Sequential(*list(self.model.children())[:-truncate])

        self.model.eval()

    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)

    def to_device(self, device):
        """Move model to specified device and return self for chaining.
        
        Args:
            device: Target device (e.g., 'cuda' or 'cpu')
            
        Returns:
            self: The model instance
        """
        self.to(device)
        self.device = device
        return self

class ResNetRegressionNet(nn.Module):
    """ResNet-based regression network for flow prediction.
    
    A neural network that uses a truncated ResNet backbone followed by fully connected
    layers to perform regression. Designed for predicting scalar values from image inputs.
    
    Attributes:
        input_shape (tuple): Expected input shape (channels, height, width)
        transforms (list): List of transforms to apply to inputs
        resnetbody (nn.Module): Truncated ResNet backbone
        avgpool (nn.Module): Adaptive average pooling layer
        fclayers (nn.Module): Fully connected layers for regression
    """

    def __init__(
        self,
        input_shape=(3, 384, 512),
        aux_input_size=0,
        aux_model=None,
        aux_encoder_layers=[64, 32],
        aux_encoder_dropout=0.0,
        aux_lstm_hidden=64,
        aux_lstm_layers=2,
        aux_lstm_dropout=0.0,
        transforms=[],
        resnet_size=18,
        truncate=2,
        num_hlayers=[256, 64],
        dropout_rate=0.0,
    ):
        """Initialize the regression network.
        
        Args:
            input_shape (tuple): Expected input shape (channels, height, width)
            aux_input_size (int): Size of auxiliary input features
            aux_model (str): Type of auxiliary data processing ('concat', 'encoder', or 'lstm')
            aux_encoder_layers (list): List of hidden layer sizes for auxiliary encoder
            aux_encoder_dropout (float): Dropout rate for auxiliary encoder layers
            aux_lstm_hidden (int): Hidden size for LSTM auxiliary encoder
            aux_lstm_layers (int): Number of LSTM layers
            aux_lstm_dropout (float): Dropout rate for LSTM layers
            transforms (list): List of transforms to apply to inputs
            resnet_size (int): Size of ResNet backbone
            truncate (int): Number of layers to remove from ResNet
            num_hlayers (list): List of hidden layer sizes for final layers
            dropout_rate (float): Dropout rate for fully connected layers
        """
        super(ResNetRegressionNet, self).__init__()
        self.input_shape = input_shape
        self.input_nchannels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.transforms = transforms
        self.aux_input_size = aux_input_size
        self.aux_model = aux_model
        
        # Initialize ResNet backbone
        self.resnetbody = ResNet(size=resnet_size, truncate=truncate)
            
        # Get number of features from ResNet
        num_resnet_features = get_output_shape(self.resnetbody, input_shape=(1, *input_shape))[1]
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Initialize auxiliary processing
        if aux_input_size > 0 and aux_model is not None:
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
            elif aux_model == "lstm":
                logger.info(f"Using LSTM auxiliary encoder with input size {aux_input_size}, hidden size {aux_lstm_hidden}, layers {aux_lstm_layers}, and dropout {aux_lstm_dropout}")
                self.aux_encoder = nn.LSTM(
                    input_size=aux_input_size,
                    hidden_size=aux_lstm_hidden,
                    num_layers=aux_lstm_layers,
                    dropout=aux_lstm_dropout if aux_lstm_layers > 1 else 0,
                    batch_first=True
                )
                aux_output_size = aux_lstm_hidden
            elif aux_model == "concat":
                logger.info(f"Using auxiliary concat model with input size {aux_input_size}")
                self.aux_encoder = None
                aux_output_size = aux_input_size
            else:
                raise ValueError(f"Invalid auxiliary model: {aux_model}")
        else:
            logger.info("No auxiliary model")
            self.aux_encoder = None
            aux_output_size = 0

        # Build fully connected layers with optional dropout
        self.fclayer_modules = []
        combined_input_size = num_resnet_features + aux_output_size
        
        for i, (in_features, out_features) in enumerate(zip([combined_input_size] + num_hlayers[:-1], num_hlayers)):
            self.fclayer_modules.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU()
            ])
            if dropout_rate > 0:
                self.fclayer_modules.append(nn.Dropout(dropout_rate))
        self.fclayer_modules.extend([nn.Linear(num_hlayers[-1], 1)])

        self.fclayers = nn.Sequential(*self.fclayer_modules)

    def forward_single(self, x, aux=None):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            aux (torch.Tensor, optional): Auxiliary features tensor. For LSTM, shape should be
                (batch_size, sequence_length, features)
            
        Returns:
            torch.Tensor: Predicted scalar value for each input in the batch
        """
        x = self.resnetbody(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Process auxiliary features if provided
        if aux is not None and self.aux_input_size > 0 and self.aux_model is not None:
            if self.aux_model == "lstm":
                # Process sequence through LSTM
                aux_out, _ = self.aux_encoder(aux)
                # Take the last output
                aux = aux_out[:, -1, :]
            elif self.aux_model == "encoder":
                aux = self.aux_encoder(aux)
            # For concat model, use aux as is
            x = torch.cat([x, aux], dim=1)
            
        x = self.fclayers(x)
        return x.squeeze()

    def forward(self, x, aux=None):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            aux (torch.Tensor, optional): Auxiliary features tensor of shape (batch_size, n_auxiliary)
            
        Returns:
            torch.Tensor: Predicted scalar value for each input in the batch
        """
        return self.forward_single(x, aux)

    def get_features(self, x, aux=None):
        """Extract features before the final layer."""
        x = self.resnetbody(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if aux is not None and self.aux_input_size > 0:
            if self.aux_model == "lstm":
                # Process sequence through LSTM
                aux_out, _ = self.aux_encoder(aux)
                # Take the last output
                aux = aux_out[:, -1, :]
            elif self.aux_model == "encoder":
                aux = self.aux_encoder(aux)
            x = torch.cat([x, aux], dim=1)
        
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
            'input_shape': self.input_shape,
            'aux_input_size': self.aux_input_size,
            'aux_model': self.aux_model
        }

class ResNetRankNet(ResNetRegressionNet):
    """ResNet-based network for learning to rank images with auxiliary data.
    
    Extends ResNetRegressionNet to handle pairs of images and auxiliary data for learning to rank.
    The network produces scalar scores for each image that can be compared to
    determine relative ranking.
    
    Inherits all attributes from ResNetRegressionNet.
    """

    def __init__(
        self,
        input_shape=(3, 384, 512),
        aux_input_size=0,
        transforms=[],
        resnet_size=50,
        truncate=2,
        num_hlayers=[256, 64],
        dropout_rate=0.0,
        aux_model=None,
        aux_encoder_layers=[64, 32],
        aux_encoder_dropout=0.0,
        aux_lstm_hidden=64,
        aux_lstm_layers=2,
        aux_lstm_dropout=0.0,
    ):
        """Initialize the ranking network.
        
        Args:
            input_shape (tuple): Expected input shape (channels, height, width)
            aux_input_size (int): Size of auxiliary input features
            transforms (list): List of transforms to apply to inputs
            resnet_size (int): Size of ResNet backbone
            truncate (int): Number of layers to remove from ResNet
            num_hlayers (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate for fully connected layers
            aux_model: Type of auxiliary data processing ('concat' or 'encoder')
            aux_encoder_layers: List of hidden layer sizes for auxiliary encoder
            aux_encoder_dropout: Dropout rate for auxiliary encoder layers
            aux_lstm_hidden: Hidden size for LSTM auxiliary encoder
            aux_lstm_layers: Number of LSTM layers
            aux_lstm_dropout: Dropout rate for LSTM layers
        """
        super().__init__(
            input_shape=input_shape,
            aux_input_size=aux_input_size,
            aux_model=aux_model,
            aux_encoder_layers=aux_encoder_layers,
            aux_encoder_dropout=aux_encoder_dropout,
            transforms=transforms,
            resnet_size=resnet_size,
            truncate=truncate,
            num_hlayers=num_hlayers,
            dropout_rate=dropout_rate,
            aux_lstm_hidden=aux_lstm_hidden,
            aux_lstm_layers=aux_lstm_layers,
            aux_lstm_dropout=aux_lstm_dropout,
        )
        
    def forward(self, input1, input2, aux1=None, aux2=None):
        """Forward pass for a pair of inputs.
        
        Args:
            input1 (torch.Tensor): First input tensor
            input2 (torch.Tensor): Second input tensor
            aux1 (torch.Tensor, optional): First auxiliary input
            aux2 (torch.Tensor, optional): Second auxiliary input
            
        Returns:
            tuple: Predicted scalar scores for both inputs
        """
        output1 = self.forward_single(input1, aux1)
        output2 = self.forward_single(input2, aux2)
        return output1, output2

def get_output_shape(model, input_shape=(1, 3, 224, 224)):
    x = torch.randn(*input_shape)
    out = model(x)
    return out.shape

def fit(model, criterion, optimizer, train_dl, device, epoch_num=None, verbose=False):
    """Train model for one epoch."""
    model.train()
    batch_loss_logger = MetricLogger()
    batch_time_logger = MetricLogger()

    for bidx, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
        batch_starttime = time.time()

        if isinstance(criterion, (torch.nn.MarginRankingLoss, RankNetLoss)):
            # Check if we have auxiliary data
            if len(batch) == 5:  # With aux data
                inputs1, inputs2, aux1, aux2, labels = batch
                if next(model.parameters()).is_cuda:
                    inputs1 = inputs1.to(device)
                    inputs2 = inputs2.to(device)
                    aux1 = aux1.to(device)
                    aux2 = aux2.to(device)
                    labels = labels.to(device)
                optimizer.zero_grad()
                outputs1, outputs2 = model(inputs1, inputs2, aux1, aux2)
            else:  # Without aux data
                inputs1, inputs2, labels = batch
                if next(model.parameters()).is_cuda:
                    inputs1 = inputs1.to(device)
                    inputs2 = inputs2.to(device)
                    labels = labels.to(device)
                optimizer.zero_grad()
                outputs1, outputs2 = model(inputs1, inputs2)
                
            loss = criterion(outputs1, outputs2, labels)
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
                    if "outputs1" not in model_outputs.keys():
                        # Check if we have auxiliary data
                        if len(batch) == 5:  # With aux data
                            inputs1, inputs2, aux1, aux2, labels = batch
                            if next(model.parameters()).is_cuda:
                                inputs1 = inputs1.to(device)
                                inputs2 = inputs2.to(device)
                                aux1 = aux1.to(device)
                                aux2 = aux2.to(device)
                                labels = labels.to(device)
                            outputs1, outputs2 = model(inputs1, inputs2, aux1, aux2)
                        else:  # Without aux data
                            inputs1, inputs2, labels = batch
                            if next(model.parameters()).is_cuda:
                                inputs1 = inputs1.to(device)
                                inputs2 = inputs2.to(device)
                                labels = labels.to(device)
                            outputs1, outputs2 = model(inputs1, inputs2)
                            
                        model_outputs["outputs1"] = outputs1
                        model_outputs["outputs2"] = outputs2
                    else:
                        outputs1, outputs2 = model_outputs["outputs1"], model_outputs["outputs2"]
                    cval = c(outputs1, outputs2, labels)
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
    resnet_size: int = 18,
    truncate: int = 2,
    dropout_rate: float = 0.0,
    aux_input_size: int = 0,
    aux_model: str = "concat",
    aux_encoder_layers: List[int] = None,
    aux_encoder_dropout: float = 0.0,
    aux_lstm_hidden: int = 64,
    aux_lstm_layers: int = 2,
    aux_lstm_dropout: float = 0.0,
) -> nn.Module:
    """Initialize and configure the model.
    
    Args:
        input_shape: Model input dimensions (channels, height, width)
        transforms: Dictionary of image transforms
        device: Target device for model
        resnet_size: Size of ResNet backbone (18, 34, 50, 101, or 152)
        truncate: Number of ResNet layers to truncate
        dropout_rate: Dropout rate for fully connected layers
        aux_input_size: Size of auxiliary input features
        aux_model: Type of auxiliary data processing ('concat', 'encoder', or 'lstm')
        aux_encoder_layers: List of hidden layer sizes for auxiliary encoder
        aux_encoder_dropout: Dropout rate for auxiliary encoder layers
        aux_lstm_hidden: Hidden size for LSTM auxiliary encoder
        aux_lstm_layers: Number of LSTM layers
        aux_lstm_dropout: Dropout rate for LSTM layers
    
    Returns:
        Configured model
    """
    if aux_encoder_layers is None:
        aux_encoder_layers = [64, 32]

    model = ResNetRankNet(
        input_shape=input_shape,
        transforms=transforms,
        resnet_size=resnet_size,
        truncate=truncate,
        dropout_rate=dropout_rate,
        aux_input_size=aux_input_size,
        aux_model=aux_model,
        aux_encoder_layers=aux_encoder_layers,
        aux_encoder_dropout=aux_encoder_dropout,
        aux_lstm_hidden=aux_lstm_hidden,
        aux_lstm_layers=aux_lstm_layers,
        aux_lstm_dropout=aux_lstm_dropout,
    )

    # Freeze resnet backbone initially
    for p in list(model.children())[0].parameters():
        p.requires_grad = False

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

    # set_seeds(args.random_seed)

    # Load and split data
    logger.info("Loading dataset from %s", Path(args.data_dir) / args.pairs_file)
    pairs_df = load_pairs_from_csv(Path(args.data_dir) / args.pairs_file, args.aux_timestep)
    train_df = pairs_df[pairs_df['split'] == "train"]
    val_df = pairs_df[pairs_df['split'] == "val"]
    logger.info("Dataset loaded - Train: %d samples, Validation: %d samples", 
                len(train_df), len(val_df))

    # Load auxiliary data if specified
    aux_data = None
    if args.aux_file:
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
    else:
        logger.info("No auxiliary data provided")
        aux_input_size = 0

    # Setup datasets and compute image stats
    logger.info("Computing image statistics from %d samples", args.transform_normalize_n)
    train_ds = FlowPhotoRankingPairsDataset(
        train_df, 
        args.images_dir,
        aux_data=aux_data,
        aux_model=args.aux_model,
        aux_sequence_length=args.aux_lstm_sequence_length,
        aux_timestep=args.aux_timestep,
    )
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
        aux_data=aux_data,
        transform=transforms["eval"],
        aux_model=args.aux_model,
        aux_sequence_length=args.aux_lstm_sequence_length,
        aux_timestep=args.aux_timestep,
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

    # Initialize model with auxiliary input size
    model = setup_model(
        (3, input_shape[0], input_shape[1]),
        transforms,
        device,
        resnet_size=args.resnet_size,
        truncate=args.truncate,
        dropout_rate=args.dropout_rate,
        aux_input_size=aux_input_size,
        aux_model=args.aux_model,
        aux_encoder_layers=args.aux_encoder_layers,
        aux_encoder_dropout=args.aux_encoder_dropout,
        aux_lstm_hidden=args.aux_lstm_hidden,
        aux_lstm_layers=args.aux_lstm_layers,
        aux_lstm_dropout=args.aux_lstm_dropout,
    )

    # Load from checkpoint if specified
    start_epoch = 0
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Get model summary for logging
    model_summary = model.get_model_summary()
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

    logger.info("Training completed")

    # test model
    logger.info("Testing model")

    # Load best model for testing
    best_model_path = Path(args.checkpoint_dir) / f"epoch_{best_epoch:02d}.pth"
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
    

    test_ds = FlowPhotoDataset(
        images_df, 
        args.images_dir,
        transform=transforms['eval'],  # Apply transforms consistently
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
        for batch in tqdm(test_dl, desc="Generating predictions"):
            # Handle both regular and auxiliary data cases
            if aux_data is not None:
                images, aux_features, _ = batch
                images = images.to(device)
                aux_features = aux_features.to(device)
                outputs = model.forward_single(images, aux_features)
            else:
                images, _ = batch
                images = images.to(device)
                outputs = model.forward_single(images)
                
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
    
    parser.add_argument("--model-dir", type=str, default='/opt/ml/model')
    parser.add_argument("--checkpoint-dir", type=str, default='/opt/ml/checkpoints')
    parser.add_argument("--output-dir", type=str, default='/opt/ml/output')
    parser.add_argument("--images-dir", type=str, default='/opt/ml/input/data/images')
    parser.add_argument("--data-dir", type=str, default='/opt/ml/input/data/data')

    parser.add_argument("--num-workers", type=int, default=4, help="number of data loader workers")

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
    parser.add_argument(
        "--eval-batch-size", type=int, default=None,
        help="batch size for validation (defaults to training batch size)"
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

    # input files
    parser.add_argument(
        "--pairs-file", type=str, default="pairs.csv",
        help="filename of CSV file with annotated image pairs",
    )
    parser.add_argument(
        "--images-file", type=str, default="images.csv",
        help="filename of CSV file with image table",
    )
    parser.add_argument(
        "--aux-file", type=str, default=None,
        help="filename of CSV file with auxiliary timeseries data",
    )

    # config file argument
    parser.add_argument(
        "--config", type=str, default="config.yml",
        help="path to YAML configuration file (default: config.yml)"
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
    
    # Auxiliary data model parameters
    parser.add_argument(
        "--aux-model", type=str, default=None,
        help="Type of auxiliary data processing: 'concat' for direct concatenation, 'encoder' for learned encoding, 'lstm' for sequential processing, or None for no auxiliary processing"
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

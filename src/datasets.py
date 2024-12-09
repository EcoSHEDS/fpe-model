import os
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
from PIL import Image, ImageStat
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.io import read_image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import torch

class FlowPhotoDataset(Dataset):
    """
    Args:
        table (_type_): _description_
        data_dir (_type_): _description_
        col_image_id (_type_): _description_
        col_timestamp (_type_): _description_
        col_filename (_type_): _description_
        col_label (_type_): _description_
        transform (_type_): _description_
    """

    def __init__(
        self,
        table,
        data_dir,
        col_image_id="image_id",
        col_timestamp="timestamp",
        col_filename="filename",
        col_label="value",
        transform=None,
    ) -> None:
        self.table = table
        self.data_dir = data_dir
        self.cols = {
            "filename": col_filename,
            "label": col_label,
            "timestamp": col_timestamp,
            "image_id": col_image_id,
        }
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def get_image(self, index):
        filename = self.table.iloc[index][self.cols["filename"]]
        image_path = os.path.join(self.data_dir, filename)
        try:
            image = read_image(image_path)
            image = image / 255.0  # convert to float in [0,1]
            return image
        except:
            print(f"Could not read image index {index} ({image_path})")

    def __getitem__(self, index) -> Tuple:
        image = self.get_image(index)
        label = self.table.iloc[index][self.cols["label"]]
        if self.transform:
            image = self.transform(image)
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
            # stat = PILImageStat.Stat(image)
            # means += np.array(stat.mean) / 255.0
            # stds += np.array(stat.stddev) / 255.0
        means = means / sample_size
        stds = stds / sample_size
        return means, stds

    def set_mean_std(self, means, stds):
        self.means = means
        self.stds = stds

class FlowPhotoDatasetWithAuxiliary(Dataset):
    """Dataset for single images with auxiliary features."""
    
    def __init__(
        self,
        table: pd.DataFrame,
        data_dir: str,
        auxiliary_features: List[str],
        col_image_id="image_id",
        col_timestamp="timestamp",
        col_filename="filename",
        col_label="value",
        transform: Optional[Compose] = None
    ) -> None:
        """Initialize dataset.
        
        Args:
            table: DataFrame with image information
            data_dir: Directory containing images
            auxiliary_features: List of auxiliary feature column names
            col_filename: Name of the filename column in the DataFrame
            transform: Optional transform to apply to images
        """
        self.table = table
        self.data_dir = Path(data_dir)
        self.cols = {
            "filename": col_filename,
            "label": col_label,
            "timestamp": col_timestamp,
            "image_id": col_image_id,
        }
        self.auxiliary_features = auxiliary_features or []  # Handle None case
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def get_image(self, index: int) -> torch.Tensor:
        """Load image from disk and apply initial processing."""
        filename = self.table.iloc[index][self.cols["filename"]]
        image_path = os.path.join(self.data_dir, filename)
        try:
            image = read_image(image_path)
            image = image / 255.0  # convert to float in [0,1]
            return image
        except:
            print(f"Could not read image index {index} ({image_path})")
            raise

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get an item from the dataset.
        
        Returns:
            Tuple of (image tensor, auxiliary features tensor, label)
        """
        # Get image
        image = self.get_image(idx)
        label = self.table.iloc[idx][self.cols["label"]]

        if self.transform:
            image = self.transform(image)
        
        # If no auxiliary features, return minimal tuple
        if not self.auxiliary_features:
            return image, torch.tensor([]), label
            
        # Get auxiliary features
        row = self.table.iloc[idx]
        auxiliary = torch.tensor([
            float(row[feat]) for feat in self.auxiliary_features
        ], dtype=torch.float32)
            
        return image, auxiliary, label

    def compute_mean_std(self, n: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
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
        transform=None,
    ) -> None:
        self.table = table
        self.images_dir = images_dir
        self.transform = transform

    def get_image(self, filename):
        image_path = os.path.join(self.images_dir, filename)

        try:
            image = read_image(image_path)
            image = image / 255.0  # convert to float in [0,1]
            return image
        except:
            print(f"Could not read image ({filename})")
            raise

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

        return image1, image2, label

class FlowPhotoRankingPairsDatasetWithAuxiliary(FlowPhotoRankingPairsDataset):
    """Dataset class that handles both images and auxiliary features."""
    
    def __init__(self, pairs_df, images_dir, auxiliary_features=None, transform=None):
        super().__init__(pairs_df, images_dir, transform)
        self.auxiliary_features = auxiliary_features or []
        
    def __getitem__(self, idx):
        pair = self.get_pair(idx)
        
        # Get images
        img1 = self.get_image(pair["filename_1"])
        img2 = self.get_image(pair["filename_2"])
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        # Get auxiliary features
        if not self.auxiliary_features:
            # If no auxiliary features, return just images and label
            aux1 = torch.tensor([], dtype=torch.float32)
            aux2 = torch.tensor([], dtype=torch.float32)
        else:            
            aux1 = torch.tensor([float(pair[f"{feat}_1"]) for feat in self.auxiliary_features])
            aux2 = torch.tensor([float(pair[f"{feat}_2"]) for feat in self.auxiliary_features])
            
        return img1, img2, aux1, aux2, pair["label"]

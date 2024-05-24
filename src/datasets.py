import json
import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm


class FPEDataset(Dataset):
    """
    A PyTorch Dataset class for handling USGS Flow Photo Explorer (FPE) datasets.

    This class assumes that the `root` directory contains the following:
    - A CSV file named `images.csv` (or another name if specified) that contains metadata about the images.
      This file should have columns for image IDs, timestamps, filenames, and labels.
    - A JSON file named `station.json` (or another name if specified) that contains metadata about the station.
    - A directory of images, with the image filenames matching those in the `images.csv` file. The filenames in
      the `images.csv` file should be relative to the `root` directory.
    """

    def __init__(
        self,
        root: str,
        data_file: str = "images.csv",
        station_file: str = "station.json",
        col_timestamp: str = "timestamp",
        col_filename: str = "filename",
        col_label: str = "value",
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            root (str): Root directory where images are downloaded to.
            data_file (str): Filename for the images file. Default 'images.csv'.
            station_file (str): Filename for station metadata. Default 'station.json'.
            col_timestamp (str): Column name for image timestamps. Default 'timestamp'.
            col_filename (str): Column name for image filenames. Default 'filename'.
            col_label (str): Column name for image labels. Default 'value'.
            transform (callable, optional): A function/transform that takes in an image
                and returns a transformed version. E.g, `transforms.ToTensor`
            label_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
        """
        self.root = root
        self.data = pd.read_csv(os.path.join(root, data_file))
        with open(os.path.join(root, station_file)) as f:
            self.station = json.load(f)
        self.col_timestamp = col_timestamp
        self.col_filename = col_filename
        self.col_label = col_label
        self.transform = transform
        self.label_transform = label_transform
        self.convert_timezone()

    def convert_timezone(self) -> None:
        """
        Convert the timestamps in the dataset to the timezone specified in the station metadata.
        """
        self.data[self.col_timestamp] = pd.to_datetime(
            self.data[self.col_timestamp]
        ).dt.tz_convert(self.station["timezone"])

    def __len__(self) -> int:
        return len(self.data)

    def get_image(self, filename: str) -> torch.Tensor:
        """Fetch the image at the given index in the dataset.

        Args:
            filename (str): The filename of the image to fetch.

        Returns:
            torch.Tensor: The image as a 3D tensor of type torch.float32 with
            shape (C, H, W), where C is the number of channels, H is the height
            of the image, and W is the width of the image. The values of the
            tensor are in the range [0, 1] and represent pixel intensities.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        image_path = os.path.join(self.root, filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = read_image(image_path).float() / 255.0
        return image

    def compute_mean_std(
        self, filename_col: Optional[str] = None, n: int = 1000
    ) -> Tuple[List[float], List[float]]:
        """
        Compute and return the average mean and standard deviation of RGB pixel values
        across a sample of images in the dataset.

        The sample size is the smaller of `n` and the total number of images in the dataset.

        Args:
            filename_col (str, optional): Column name for image filenames used to read images.
                If not provided, `self.col_filename` is used.
            n (int, optional): The desired sample size. Default is 1000.

        Returns:
            A tuple of two lists: the first list contains the average mean pixel
            values for the R, G, and B channels, and the second list contains the
            average standard deviation of pixel values for the R, G, and B channels.
        """
        if filename_col is None:
            filename_col = self.col_filename
        means = torch.zeros(3)
        stds = torch.zeros(3)
        unique_filenames = self.data[filename_col].unique()
        sample_size = min(len(unique_filenames), n)
        sample_indices = torch.randperm(len(unique_filenames))[:sample_size].tolist()
        for i in tqdm(sample_indices):
            image = self.get_image(unique_filenames[i])
            means += image.mean(dim=[1, 2])
            stds += image.std(dim=[1, 2])
        means /= sample_size
        stds /= sample_size
        return means.tolist(), stds.tolist()

    def set_mean_std(self, means: List[float], stds: List[float]) -> None:
        """
        Set the mean and standard deviation of RGB pixel values for the dataset.

        Args:
            means (List[float]): The average mean pixel values for the R, G, and B channels.
            stds (List[float]): The average standard deviation of pixel values for the R, G, and B channels.
        """
        self.means = means
        self.stds = stds

    def __getitem__(self, index: int) -> Tuple:
        """
        Fetch the image and the corresponding label at the given index.

        Args:
            index (int): The index of the image to fetch.

        Returns:
            tuple: A tuple containing the image and the label.
        """
        row = self.data.iloc[index]
        image = self.get_image(row[self.col_filename])
        label = row[self.col_label]
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label


class FPERankingPairsDataset(FPEDataset):
    """
    A PyTorch Dataset class for handling USGS Flow Photo Explorer (FPE) ranked pair datasets.

    This class assumes that the `root` directory contains the following:
    - A CSV file named `pairs.csv` (or another name if specified) that contains metadata about the image pairs.
      This file should have columns for pair IDs, timestamps, filenames, and labels.
    - A JSON file named `station.json` (or another name if specified) that contains metadata about the station.
    - A directory of images, with the image filenames matching those in the `pairs.csv` file. The filenames in
      the `pairs.csv` file should be relative to the `root` directory.
    """

    def __init__(
        self,
        root: str,
        data_file: str = "pairs.csv",
        station_file: str = "station.json",
        col_timestamp_1: str = "timestamp_1",
        col_timestamp_2: str = "timestamp_2",
        col_filename_1: str = "filename_1",
        col_filename_2: str = "filename_2",
        col_value_1: str = "value_1",
        col_value_2: str = "value_2",
        col_label: str = "rank",
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            root (str): Root directory where images are downloaded to.
            data_file (str): Filename for the annotations file. Default 'pairs.csv'.
            station_file (str): Filename for station metadata. Default 'station.json'.
            col_timestamp_1 (str): Column name for left image timestamps. Default 'timestamp_1'.
            col_timestamp_2 (str): Column name for right image timestamps. Default 'timestamp_2'.
            col_filename_1 (str): Column name for left image filenames. Default 'filename_1'.
            col_filename_2 (str): Column name for right image filenames. Default 'filename_2'.
            col_value_1 (str): Column name for left image values. Default 'value_1'.
            col_value_2 (str): Column name for right image values. Default 'value_2'.
            col_label (str): Column name for image labels. Default 'rank'.
            transform (callable, optional): A function/transform that takes in an image
                and returns a transformed version. E.g, `transforms.ToTensor`
            label_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
        """
        self.col_timestamp_1 = col_timestamp_1
        self.col_timestamp_2 = col_timestamp_2
        self.col_filename_1 = col_filename_1
        self.col_filename_2 = col_filename_2
        self.col_value_1 = col_value_1
        self.col_value_2 = col_value_2
        super().__init__(
            root,
            data_file=data_file,
            station_file=station_file,
            col_label=col_label,
            transform=transform,
            label_transform=label_transform,
        )

    def convert_timezone(self) -> None:
        """
        Convert the timestamps in the dataset to the timezone specified in the station metadata.
        """
        self.data[self.col_timestamp_1] = pd.to_datetime(
            self.data[self.col_timestamp_1]
        ).dt.tz_convert(self.station["timezone"])
        self.data[self.col_timestamp_2] = pd.to_datetime(
            self.data[self.col_timestamp_2]
        ).dt.tz_convert(self.station["timezone"])

    def __getitem__(self, index: int) -> Tuple:
        """
        Fetch the pair of images and the corresponding label at the given index.

        This method is overridden from the parent class to handle pairs of images.
        Instead of returning a single image and a label, it returns a pair of
        images and a label.

        Args:
            index (int): The index of the pair to fetch.

        Returns:
            tuple: A tuple containing the left image, the right image, and the
            label.
        """
        row = self.data.iloc[index]
        left_image = self.get_image(row[self.col_filename_1])
        right_image = self.get_image(row[self.col_filename_2])
        label = row[self.col_label]

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        if self.label_transform:
            label = self.label_transform(label)

        return left_image, right_image, label

    def compute_mean_std(self, n: int = 1000) -> Tuple[List[float], List[float]]:
        """
        Compute and return the average mean and standard deviation of RGB pixel values
        across a sample of images in the dataset.

        The sample size is the smaller of `n` and the total number of images in the dataset.

        Args:
            n (int, optional): The desired sample size. Default is 1000.

        Returns:
            A tuple of two lists: the first list contains the average mean pixel
            values for the R, G, and B channels, and the second list contains the
            average standard deviation of pixel values for the R, G, and B channels.
        """
        return super().compute_mean_std(self.col_filename_1, n)

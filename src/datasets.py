import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from itertools import combinations
from PIL import Image, ImageStat
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit


# class FlowPhotoTable(object):
#     def __init__(
#         self,
#         filepath,
#         image_dir,
#         col_image="filename",
#         col_flow="flow_cfs",
#         col_timestamp="timestamp",
#         tz=None,
#     ):
#         self.data = pd.read_csv(filepath, dtype={col_flow: np.float32})
#         self.data[col_timestamp] = pd.to_datetime(self.data[col_timestamp])
#         self.image_dir = image_dir
#         self.col_image = col_image
#         self.col_flow = col_flow
#         self.col_timestamp = col_timestamp
#         self.tz = tz
#         if self.tz:
#             self.data[col_timestamp] = self.data[col_timestamp].dt.tz_convert(tz=tz)

#     def filter_by_hour(self, min_hour=7, max_hour=18):
#         self.data = self.data[
#             self.data[self.col_timestamp].dt.hour.between(min_hour, max_hour)
#         ]

#     def filter_by_month(self, min_month=4, max_month=11):
#         self.data = self.data[
#             self.data[self.col_timestamp].dt.month.between(min_month, max_month)
#         ]

#     def filter_by_date(self, start_date, end_date, mode="exclude"):
#         if mode == "exclude":
#             before_start_date = self.data[self.col_timestamp] < start_date
#             after_end_date = self.data[self.col_timestamp] > end_date
#             outside_two_dates = before_start_date | after_end_date
#             filtered_dates = self.data.loc[outside_two_dates].copy()
#             self.data = filtered_dates
#         else:
#             raise NotImplementedError(
#                 'Please select "exclude" mode and provide date range to exclude.'
#             )


class FlowPhotoDataset(Dataset):
    """_summary_

    Args:
        table (_type_): _description_
        data_dir (_type_): _description_
        col_filename (_type_): _description_
        col_label (_type_): _description_
        transform (_type_): _description_
        label_transform (_type_): _description_
    """

    def __init__(
        self,
        table,
        data_dir,
        col_filename="filename",
        col_label="flow_cfs",
        transform=None,
        label_transform=None,
    ) -> None:
        self.table = table
        self.data_dir = data_dir
        self.col_filename = col_filename
        self.col_label = col_label
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self) -> int:
        return len(self.table)

    def get_image(self, index):
        filename = self.table.iloc[index][self.col_filename]
        image_path = os.path.join(
            self.data_dir,
            # "images",
            filename,
        )  # TODO: see if we can avoid secret subdirs
        try:
            image = read_image(image_path)
            image = image / 255.0  # convert to float in [0,1]
            return image
        except:
            print(f"Could not read image index {index} ({image_path})")

    def __getitem__(self, index) -> Tuple:
        image = self.get_image(index)
        label = self.table.iloc[index][self.col_label]
        if self.transform:
            image = self.transform(
                image
            )  # without converting to numpy or reading as PILImage, transforms don't work
        if self.label_transform:
            label = self.label_transform(label)
        return image, label

    def compute_mean_std(self):
        """Compute RGB channel means and stds for image samples in the dataset."""
        means = np.zeros((3))
        stds = np.zeros((3))
        sample_size = min(len(self.table), 1000)
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


class FlowPhotoRankingDataset(FlowPhotoDataset):
    def __init__(
        self,
        table,
        data_dir,
        col_filename="filename",
        col_label="flow_cfs",
        transform=None,
        label_transform=None,
    ) -> None:
        super().__init__(
            table, data_dir, col_filename, col_label, transform, label_transform
        )
        self.image_pair_sampler = None
        self.ranked_image_pairs = []

    def __len__(self):
        return len(self.ranked_image_pairs)

    def __getitem__(self, idx):
        idx1, idx2, label = self.ranked_image_pairs[idx]
        image1 = self.get_image(idx1)
        image2 = self.get_image(idx2)

        if self.transform:
            image1 = self.transform(
                image1
            )  # without converting to numpy or reading as PILImage, transforms don't work
            image2 = self.transform(
                image2
            )  # without converting to numpy or reading as PILImage, transforms don't work
        if self.label_transform:
            label = self.label_transform(label)
        return image1, image2, label

    def rank_image_pairs(self, pair_sampling_fn, num_pairs, margin=0, mode="relative"):
        self.image_pair_sampler = ImagePairSampler(self.table, pair_sampling_fn)
        self.sampled_image_pairs = self.image_pair_sampler.get_pairs(num_pairs)
        labeled_sampled_image_pairs = self.label_image_pairs(
            self.sampled_image_pairs, margin=margin, mode=mode
        )
        self.ranked_image_pairs.extend(labeled_sampled_image_pairs)

    def label_image_pairs(self, image_pairs, margin, mode="relative"):
        labeled_ranked_image_pairs = []
        for i in range(len(image_pairs)):
            idx1 = image_pairs[i][0]
            idx2 = image_pairs[i][1]
            if mode == "absolute":
                if (
                    self.table[self.col_label].iloc[idx1]
                    - self.table[self.col_label].iloc[idx2]
                ) > margin:
                    # first idx has higher discharge
                    label = 1
                elif (
                    self.table[self.col_label].iloc[idx1]
                    - self.table[self.col_label].iloc[idx2]
                ) < -margin:
                    # first idx has lower discharge
                    label = -1
                else:
                    # both indices have similar discharge
                    label = 0
            elif mode == "relative":
                left_disch = self.table[self.col_label].iloc[idx1]
                right_dishc = self.table[self.col_label].iloc[idx2]
                min_disch = min(left_disch, right_dishc)
                if (left_disch - right_dishc) / min_disch > margin:
                    # first idx has higher discharge
                    label = 1
                elif (left_disch - right_dishc) / min_disch < -margin:
                    # first idx has lower discharge
                    label = -1
                else:
                    # both indices have similar discharge
                    label = 0
            else:
                raise NotImplementedError(
                    "The only valid values for mode are 'relative' and 'absolute'."
                )
            labeled_ranked_image_pairs.extend(
                [(idx1, idx2, label), (idx2, idx1, -1 * label)]
            )
        return labeled_ranked_image_pairs


class ImagePairSampler:
    """Sampling methods for pairs of streamflow images."""

    def __init__(self, table, sampling_fn):
        self.table = table
        self.pair_sampling_fn = sampling_fn

    def get_pairs(self, num_pairs):
        return self.pair_sampling_fn(self.table, num_pairs)


def random_pairs(table, num_pairs):
    # each image is paired with another image at random
    all_img_idx_combinations = list(combinations(range(len(table)), 2))
    num_combinations = len(all_img_idx_combinations)
    if len(all_img_idx_combinations) < num_pairs:
        combinations_sample = [
            all_img_idx_combinations[i]
            for i in np.random.choice(range(num_combinations), num_pairs, replace=True)
        ]
    else:
        combinations_sample = [
            all_img_idx_combinations[i]
            for i in np.random.choice(range(num_combinations), num_pairs, replace=False)
        ]
    return combinations_sample


def temporally_adjacent_pairs(table, num_pairs):
    # each image is paired with the one immediately adjacent by timestamp
    timestamps = table.DATETIME.values
    sorted_timestamps_idx = np.argsort(timestamps)
    img_idx_combinations = [
        (sorted_timestamps_idx[i], sorted_timestamps_idx[i + 1])
        for i in range(len(sorted_timestamps_idx) - 1)
    ]
    num_combinations = len(img_idx_combinations)
    if len(img_idx_combinations) < num_pairs:
        combinations_sample = [
            img_idx_combinations[i]
            for i in np.random.choice(range(num_combinations), num_pairs, replace=True)
        ]
    else:
        combinations_sample = [
            img_idx_combinations[i]
            for i in np.random.choice(range(num_combinations), num_pairs, replace=False)
        ]
    return combinations_sample


def temporally_distributed_pairs(table, num_pairs):
    # image pairs are sampled such that the time deltas in the pairs are spread
    timestamps = table.DATETIME.values
    sorted_timestamps_idx = np.argsort(timestamps)
    max_stepsize = min(720, int(0.5 * len(timestamps)))
    stepsizes = np.concatenate(
        [np.arange(1, 24), np.arange(24, 96, 12), np.arange(90, max_stepsize, 12)]
    )  # np.arange(1, int(0.5*len(timestamps)), 6)
    total_samples_per_stepsize = [len(timestamps) - ss for ss in stepsizes]
    sample_per_stepsize = min(total_samples_per_stepsize)
    img_idx_combinations = []
    for stepsize in stepsizes:
        stepsize_pairs = [
            (sorted_timestamps_idx[i], sorted_timestamps_idx[i + stepsize])
            for i in range(len(sorted_timestamps_idx) - stepsize)
        ]
        sampled_stepsize_pairs = [
            stepsize_pairs[i]
            for i in np.random.choice(
                range(len(stepsize_pairs)), sample_per_stepsize, replace=False
            )
        ]
        img_idx_combinations.extend(sampled_stepsize_pairs)
    num_combinations = len(img_idx_combinations)
    if len(img_idx_combinations) < num_pairs:
        combinations_sample = [
            img_idx_combinations[i]
            for i in np.random.choice(range(num_combinations), num_pairs, replace=True)
        ]
    else:
        combinations_sample = [
            img_idx_combinations[i]
            for i in np.random.choice(range(num_combinations), num_pairs, replace=False)
        ]
    return combinations_sample


def discharge_distributed_pairs(table, num_pairs):
    discharge_diffs = np.abs(
        np.subtract.outer(table.DISCHARGE.tolist(), table.DISCHARGE.tolist())
    )
    for i in range(discharge_diffs.shape[0]):
        for j in range(0, i + 1):
            discharge_diffs[i, j] = -1
    discharge_diffs_vals = [
        discharge_diffs[i, j]
        for i in range(discharge_diffs.shape[0])
        for j in range(i + 1, discharge_diffs.shape[0])
    ]
    num_discharge_bins = 20
    discharge_diffs_quantiles = np.linspace(0, 1, num_discharge_bins + 1, endpoint=True)
    discharge_diffs_qvals = np.quantile(discharge_diffs_vals, discharge_diffs_quantiles)
    num_per_range = int(np.ceil(num_pairs / num_discharge_bins))
    combinations_sample = []
    for binidx in range(num_discharge_bins):
        bin_l = discharge_diffs_qvals[binidx]
        bin_u = discharge_diffs_qvals[binidx + 1]
        r, c = np.where(
            np.logical_and(discharge_diffs >= bin_l, discharge_diffs < bin_u)
        )
        discharge_diff_in_range = list(zip(r, c))
        sampled_pairs = [
            discharge_diff_in_range[si]
            for si in np.random.choice(
                range(len(discharge_diff_in_range)), num_per_range, replace=False
            )
        ]
        combinations_sample.extend(sampled_pairs)
    if len(combinations_sample) > num_pairs:
        combinations_sample = [
            combinations_sample[si]
            for si in np.random.choice(
                range(len(combinations_sample)), num_pairs, replace=False
            )
        ]
    return combinations_sample


def classify3(low_value, high_value, value):
    if value <= low_value:
        return "low"
    elif value >= high_value:
        return "high"
    else:
        return "med"


class DatasetSplitter(object):
    """Splitters split Datasets into train/validation/test sets."""

    def split(self, dataset, frac_train, frac_val, frac_test) -> Dict:
        np.testing.assert_almost_equal(frac_train + frac_val + frac_test, 1.0)
        num_datapoints = len(dataset)
        train_cutoff = int(frac_train * num_datapoints)
        val_cutoff = int((frac_train + frac_val) * num_datapoints)
        indices = np.arange(num_datapoints)
        train_indices = indices[:train_cutoff]
        val_indices = indices[train_cutoff:val_cutoff]
        test_indices = indices[val_cutoff:]
        return {
            "train": dataset.iloc[train_indices],
            "val": dataset.iloc[val_indices],
            "test": dataset.iloc[test_indices],
        }


class RandomStratifiedWeeklyFlow(DatasetSplitter):
    def split(self, dataset, frac_train, frac_val, frac_test, seed=1) -> Dict:
        np.testing.assert_almost_equal(frac_train + frac_val + frac_test, 1.0)

        df = dataset.copy()
        df.sort_values(by="timestamp", inplace=True, ignore_index=True)
        df["week"] = df["timestamp"].dt.isocalendar().week
        df["year"] = df["timestamp"].dt.isocalendar().year

        weekly_flow_means = (
            df[["flow_cfs", "year", "week"]]
            .groupby(["year", "week"])
            .mean()
            .rename(columns={"flow_cfs": "mean_flow_cfs"})
        )
        weekly_flow_quantiles = np.quantile(
            weekly_flow_means["mean_flow_cfs"].values, [0.25, 0.75], axis=0
        )
        weekly_flow_means["flow_class"] = weekly_flow_means["mean_flow_cfs"].map(
            lambda x: classify3(weekly_flow_quantiles[0], weekly_flow_quantiles[1], x)
        )
        weekly_flow_means["week_index"] = range(len(weekly_flow_means.index))

        df = (
            df.set_index(["year", "week"])
            .join(weekly_flow_means, on=["year", "week"])
            .reset_index()
        )
        weeks = weekly_flow_means.reset_index()

        X = weeks["week_index"]
        y = weeks["flow_class"]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=frac_test, random_state=seed)
        week_idx_train_val, week_idx_test = list(sss.split(X, y))[0]
        X_trv = [X[i] for i in sorted(week_idx_train_val)]
        X_t = [X[i] for i in sorted(week_idx_test)]
        y_trv = [y[i] for i in sorted(week_idx_train_val)]
        # y_t = [y[i] for i in sorted(week_idx_test)]

        rescaled_frac_val = frac_val / (1 - frac_test)
        sss_trv = StratifiedShuffleSplit(
            n_splits=1, test_size=rescaled_frac_val, random_state=seed + 1
        )
        week_idx_train, week_idx_val = list(sss_trv.split(X_trv, y_trv))[0]
        X_tr = [X_trv[i] for i in sorted(week_idx_train)]
        X_v = [X_trv[i] for i in sorted(week_idx_val)]
        # y_tr = [y_trv[i] for i in sorted(week_idx_train)]
        # y_v = [y_trv[i] for i in sorted(week_idx_val)]

        train_inds = np.where(df.week_index.isin(X_tr))[0]
        val_inds = np.where(df.week_index.isin(X_v))[0]
        test_inds = np.where(df.week_index.isin(X_t))[0]
        return {
            "train": df.iloc[train_inds],
            "val": df.iloc[val_inds],
            "test": df.iloc[test_inds],
        }


# class FlowPhotoRankingDataset(FlowPhotoDataset):
#     def __init__(
#         self,
#         table,
#         data_dir,
#         col_filename="filename",
#         col_label="flow_cfs",
#         transform=None,
#         label_transform=None,
#     ) -> None:
#         super().__init__(
#             table, data_dir, col_filename, col_label, transform, label_transform
#         )

# def compute_mean_std(self):
#     means = np.zeros((3))
#     stds = np.zeros((3))
#     sample_size = min(len(self.table), 1000)
#     sample_indices = np.random.choice(
#         len(self.table), size=sample_size, replace=False
#     )
#     for index in sample_indices:
#         image, _ = self[index]
#         image = to_pil_image(image)
#         stat = ImageStat.Stat(image)
#         means += np.array(stat.mean) / 255.0
#         stds += np.array(stat.stddev) / 255.0
#     means = means / sample_size
#     stds = stds / sample_size
#     return means, stds

# def set_mean_std(
#     self, means, stds
# ):  # TODO: explain if means are 0-1 scaled or 0-255 scaled
#     self.means = means
#     self.stds = stds


# class FlowPhotoSite:
#     def __init__(self, table, data_dir, col_filename="filename", col_label="flow_cfs"):
#         self.table = table
#         self.data_dir = data_dir
#         self.col_filename = col_filename
#         self.col_label = col_label


# class FlowPhotoDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         table,
#         data_dir,
#         col_filename="filename",
#         col_label="flow_cfs",
#         transform=None,
#         label_transform=None,
#     ):
#         self.table = table
#         self.data_dir = data_dir
#         self.col_filename = col_filename
#         self.col_label = col_label
#         self.transform = transform
#         self.label_transform = label_transform

# def __len__(self):
#     return len(self.table) # no. rows in csv file

# def __getitem__(self, idx):
#     filename = self.table.iloc[idx][self.col_filename]
#     img_path = os.path.join(self.data_dir, "images", filename)
#     image = read_image(img_path) # read image file as tensor
#     image = image / 255.0 # convert to float in [0,1]

#     label = self.table.iloc[idx][self.col_label] # get label from table
#     if self.transform:
#         image = self.transform(image) # transform image
#     if self.label_transform:
#         label = self.label_transform(label) # transform label
#     return image, label

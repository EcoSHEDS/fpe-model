import os
import logging
import time
import random
import numpy as np
import pandas as pd
import torch
from urllib.parse import urlparse
from tqdm import tqdm
from .losses import MSELoss, RankNetLoss


def log(log_file):
    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


# https://stackoverflow.com/a/47087513
def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b

def get_url_path(url):
    return urlparse(url).path[1:]

def set_seeds(seed, multigpu=False):
    random.seed(seed)
    np.random.seed(seed)

    # In general seed PyTorch operations
    torch.manual_seed(seed)
    if not multigpu:
        # If you are using CUDA on 1 GPU, seed it
        torch.cuda.manual_seed(seed)
    else:
        # If you are using CUDA on more than 1 GPU, seed them all
        torch.cuda.manual_seed_all(seed)
    # Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware.
    # torch.backends.cudnn.benchmark = False # this might be slowing down training
    # Certain operations in Cudnn are not deterministic, and this line will force them to behave!
    # torch.backends.cudnn.deterministic = True # this might be slowing down training


def load_data(data_file, col_timestamp="timestamp", col_filename="filename", col_url="url"):
    df = pd.read_csv(data_file)
    df[col_timestamp] = pd.to_datetime(df[col_timestamp])
    df[col_filename] = df[col_url].apply(get_url_path)
    df.sort_values(by=col_timestamp, inplace=True, ignore_index=True)
    return df


def convert_tz(df, tz: str, datetime_col: str = "timestamp"):
    """Convert a column in a DataFrame from one time zone to another.

    Args:
        df (pd.DataFrame): DataFrame containing a Datetime column.
        tz (str): Time zone to convert Datetime column to.
        datetime_col (str, optional): Name of the Datetime column to convert to the specified time zone. Defaults to "timestamp".

    Returns:
        pd.DataFrame: DataFrame with the Datetime column converted to the specified time zone.
    """
    df[datetime_col] = df[datetime_col].dt.tz_convert(tz=tz)
    return df


def filter_by_hour(df, min_hour=7, max_hour=18, datetime_col="timestamp"):
    """Filter DataFrame rows based on hour of day in a Datetime column.

    The hour of each Datetime value ranges from 0 to 23. Rows whose Datetime column hour values are (strictly)
    less than min_hour will be excluded. Similarly, rows whose Datetime column hour values are (strictly) greater
    thank max_hour will be excluded.

    That is, filter_by_hour with min_hour=7 and max_hour=18 returns only rows whose Datetime column is between
    7:00:00 AM and 6:59:59 PM inclusive.

    Args:
        df (pd.DataFrame): DataFrame containing a Datetime column.
        min_hour (int, optional): Minimum value for Datetime hour values. Defaults to 7.
        max_hour (int, optional): Maximum value for Datetime hour vaues. Defaults to 18.
        datetime_col (str, optional): Name of the Datetime column to use for filtering. Defaults to "timestamp".

    Returns:
        pd.DataFrame: Filtered DataFrame with entries whose Datetime hour is before min_hour or after max_hour removed.
    """
    df = df[df[datetime_col].dt.hour.between(min_hour, max_hour)]
    return df


def filter_by_month(df, min_month=4, max_month=11, col_timestamp="timestamp"):
    """Filter DataFrame rows based on month of year in a Datetime column.

    The month of each Datetime value ranges from 1 to 12. Rows whose Datetime column month values are (strictly)
    less than min_month will be excluded. Similarly, rows whose Datetime column month values are (strictly) greater
    thank max_month will be excluded.

    That is, filter_by_month with min_month=4 and max_month=11 returns only rows whose Datetime column is between
    April and November inclusive.

    Args:
        df (pd.DataFrame): DataFrame containing a Datetime column.
        min_month (int, optional): Minimum value for Datetime month values. Defaults to 4.
        max_month (int, optional): Maximum value for Datetime month vaues. Defaults to 11`.
        datetime_col (str, optional): Name of the Datetime column to use for filtering. Defaults to "timestamp".

    Returns:
        pd.DataFrame: Filtered DataFrame with entries whose Datetime month is before min_month or after max_month removed.
    """
    df = df[df[col_timestamp].dt.month.between(min_month, max_month)]
    return df


def filter_by_date(df, start_date, end_date, col_timestamp="timestamp", mode="exclude"):
    if mode == "exclude":
        before_start_date = df[col_timestamp] < start_date
        after_end_date = df[col_timestamp] > end_date
        outside_two_dates = before_start_date | after_end_date
        filtered_dates = df.loc[outside_two_dates].copy()
        df = filtered_dates
        return df
    else:
        raise NotImplementedError(
            'Please select "exclude" mode and provide date range to exclude.'
        )


def get_output_shape(model, input_shape=(1, 3, 224, 224)):
    x = torch.randn(*input_shape)
    out = model(x)
    return out.shape


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


class Accuracy(torch.nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, outputs, targets):
        _, predicted = torch.max(outputs, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum()
        return 100 * correct / float(total)


class PairwiseRankAccuracy(torch.nn.Module):
    def __init__(self):
        super(PairwiseRankAccuracy, self).__init__()

    def forward(self, outputs_i, outputs_j, targets, boundaries=[0.33, 0.66]):
        oij = outputs_i - outputs_j
        Pij = torch.sigmoid(oij)
        preds = torch.zeros_like(targets)
        preds = torch.where(Pij < boundaries[0], -1, preds)
        preds = torch.where(Pij > boundaries[1], 1, preds)
        # preds[(Pij>boundaries[1])] = 1
        # preds[(Pij<boundaries[0])] = -1
        total = targets.size(0)
        # correct = (preds == targets).sum()
        correct = torch.eq(preds, targets).sum()
        return 100 * correct / float(total)
        # zeros = (Pij>=0.33).float()*(Pij<=0.66).float()
        # target_probs = 0.5*(targets + 1)


def fit(model, criterion, optimizer, train_dl, device, epoch_num=None, verbose=False):
    """Train model for one epoch.

    Args:
        model (torch.nn.Module): network to train
        criterion (torch.nn.Module): loss function(s) used to train network weights
        optimizer (torch.optim.Optimizer): algorithm used to optimize network weights
        train_dl (torch.utils.DataLoader): data loader for training set
        epoch_num (int): epoch number for logging

    Returns:
        batch_loss_logger.avg (float): average criterion loss per batch during training
    """
    print("Training")
    model.train()  # ensure model is in train mode
    # train_dl.dataset.train()  # ensure train transforms are applied
    batch_loss_logger = MetricLogger()
    batch_time_logger = MetricLogger()

    for bidx, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
        batch_starttime = time.time()

        if isinstance(criterion, (torch.nn.MarginRankingLoss, RankNetLoss)):
            # paired inputs ->[model]-> paired outputs ->[criterion]-> value
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
        else:
            # inputs ->[model]-> outputs ->[criterion]-> value
            inputs, labels = batch
            if next(model.parameters()).is_cuda:
                inputs = inputs.to(device)
                labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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
        f"[Epoch {epoch_num}]\t{batch_time_logger.sum:.2f} s\t{batch_loss_logger.avg:.4f}"
    )

    return batch_loss_logger.avg


def validate(model, criterions, dl, device):
    """Calculate multiple criterion for a model on a dataset."""
    print("Validating")
    model.eval()
    # dl.dataset.evaluate()
    criterion_loggers = [MetricLogger() for i in range(len(criterions))]
    with torch.no_grad():  # ensure no gradients are computed
        for bidx, batch in tqdm(enumerate(dl), total=len(dl)):
            model_outputs = {}
            for i, c in enumerate(criterions):
                # start_timer = time.time()
                if isinstance(
                    c, (torch.nn.MarginRankingLoss, RankNetLoss, PairwiseRankAccuracy)
                ):
                    # paired inputs ->[model]-> paired outputs ->[criterion]-> value
                    if "outputs1" not in model_outputs.keys():
                        # store model outputs from forward pass in case another criterion needs the same
                        inputs1, inputs2, labels = batch
                        if next(model.parameters()).is_cuda:
                            inputs1, inputs2, labels = (
                                inputs1.to(device),
                                inputs2.to(device),
                                labels.to(device),
                            )
                        outputs1, outputs2 = model(inputs1, inputs2)
                        model_outputs["outputs1"] = outputs1
                        model_outputs["outputs2"] = outputs2
                    else:
                        # load previously computed model outputs from forward pass
                        outputs1, outputs2 = (
                            model_outputs["outputs1"],
                            model_outputs["outputs2"],
                        )
                    cval = c(outputs1, outputs2, labels)
                else:
                    # inputs ->[model]-> outputs ->[criterion]-> value
                    if "outputs" not in model_outputs.keys():
                        # store model outputs from forward pass in case another criterion needs the same
                        inputs, labels = batch
                        if next(model.parameters()).is_cuda:
                            inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        model_outputs["outputs"] = outputs
                    else:
                        # load previously computed model outputs from forward pass
                        outputs = model_outputs["outputs"]
                    cval = c(outputs, labels)
                criterion_loggers[i].update(cval.item())
    return [cl.avg for cl in criterion_loggers]


def evaluate_criterion(model, criterion, dl, device):
    """Calculate criterion for a model on a dataset.

    During an epoch of training, the loss criterion is
    computed for each batch of training data passed through
    the model, and the average of these losses is computed
    for the epoch.

    However, we may want to compute:
    - the loss of the (frozen) model on the train dataset
    after X epochs of training, without applying transforms
    applied to training samples only at train time
    - the loss of the (frozen) model on the validation set
    after X epochs of training, which is not otherwise
    computed during training

    """
    model.eval()  # ensure model is in eval mode
    # dl.dataset.evaluate()  # ensure eval transforms are applied
    batch_loss_logger = MetricLogger()
    with torch.no_grad():  # ensure no gradients are computed
        for bidx, batch in enumerate(dl):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_loss_logger.update(loss.item())
    return batch_loss_logger.avg


def evaluate_accuracy(model, dl, device):
    model.eval()  # ensure model is in eval mode
    # dl.dataset.evaluate()  # ensure eval transforms are applied
    correct = 0
    total = 0
    with torch.no_grad():  # ensure no gradients are computed
        for bidx, batch in enumerate(dl):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / float(total)

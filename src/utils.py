import random
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.losses import RankNetLoss


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

def load_pairs(data_file):
    df = pd.read_csv(data_file)
    df['timestamp_1'] = pd.to_datetime(df['timestamp_1'])
    df['timestamp_2'] = pd.to_datetime(df['timestamp_2'])
    return df

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
    Returns:
        batch_loss_logger.avg (float): average criterion loss per batch during training
    """
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
        f"[Epoch {epoch_num}|train]\t{batch_time_logger.sum:.2f} s\t{batch_loss_logger.avg:.4f}"
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


def parse_configargparse_args(params_file):
    """Parse parameters and their values from a configargparse output file.

    Args:
        params_file (str): Path to the file containing the configargparse output.

    Raises:
        NotImplementedError: If the params file contains variables with more than one value.

    Returns:
        dict: A dictionary containing the parsed parameter values. The keys are the parameter names
              without leading hyphens, converted to snake_case. The values are either the assigned
              values or True for standalone flags.
    """
    with open(params_file, "r") as f:
        lines = f.readlines()

    # Find the line that marks the beginning of config file arguments
    config_file_start = None
    for i, line in enumerate(lines):
        if line.startswith("Config File"):
            config_file_start = i + 1
            break

    # Find the line that marks the beginning of default arguments
    defaults_start = None
    for i, line in enumerate(lines):
        if line.startswith("Defaults:"):
            defaults_start = i + 1
            break

    # Initialize the dictionary
    parsed_values = {}

    # Process the command line arguments
    cmd_line_data = lines[0].strip("Command Line Args:").split()
    i = 0
    while i < len(cmd_line_data):
        arg = cmd_line_data[i]
        if arg.startswith("-"):
            key = arg.lstrip("-").replace("-", "_")
            if i + 1 < len(cmd_line_data) and not cmd_line_data[i + 1].startswith("-"):
                value = cmd_line_data[i + 1]
                parsed_values[key] = value
                i += 2
            else:
                parsed_values[key] = True
                i += 1
        else:
            i += 1

    # Process config file and default arguments
    if config_file_start is not None:
        remaining_args = [
            line
            for line in lines[config_file_start:]
            if line != lines[defaults_start - 1]
        ]
        for arg in remaining_args:
            var_name, *var_val = arg.split()
            if len(var_val) > 1:
                raise NotImplementedError(
                    "Params file parser currently only handles variables with 1 value (not lists)."
                )
            parsed_values[var_name[:-1].lstrip("-").replace("-", "_")] = var_val[0]

    types = {
        "c": str,
        "site": str,
        "data_file": str,
        "image_root_dir": str,
        "col_timestamp": str,
        "output_root_dir": str,
        "min_month": int,
        "max_month": int,
        "min_hour": int,
        "max_hour": int,
        "margin": float,
        "margin_mode": str,
        "num_train_pairs": int,
        "num_eval_pairs": int,
        "augment": bool,
        "normalize": bool,
        "epochs": int,
        "batch_size": int,
        "lr": float,
        "warm_start_from_checkpoint": str,
        "unfreeze_after": int,
        "random_seed": int,
        "gpu": int,
        "annotations": str,
    }
    for key, val in parsed_values.items():
        parsed_values[key] = types[key](val)
    return parsed_values


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def get_batch_creds(session, role_arn):
    sts = session.client("sts")
    response = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName=f"fpe-sagemaker-session--{timestamp()}"
    )
    return response['Credentials']

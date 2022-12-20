import argparse
import json
import logging
import os
import sys
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

#import sagemaker_containers
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.models import resnet18

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def classify3(low_value, high_value, value):
    if value <= low_value:
        return 'low'
    elif value >= high_value:
        return 'high'
    else:
        return 'med'

def split_weekly_flow(x, test_size=0.2, seed=1):
    df = x.copy()
    df['week'] = df['timestamp'].dt.isocalendar().week
    df['year'] = df['timestamp'].dt.isocalendar().year
    df.sort_values(by='timestamp', inplace=True, ignore_index=True)
    
    weekly_flow_means = df[['flow_cfs', 'year', 'week']].groupby(['year', 'week']).mean().rename(columns={'flow_cfs': 'mean_flow_cfs'})
    weekly_flow_quantiles = np.quantile(weekly_flow_means['mean_flow_cfs'].values, [.25, .75], axis=0)
    
    weekly_flow_means['flow_class'] = weekly_flow_means['mean_flow_cfs'].map(lambda x: classify3(weekly_flow_quantiles[0], weekly_flow_quantiles[1], x))
    weekly_flow_means['week_index'] = range(len(weekly_flow_means.index))

    df = df.set_index(['year', 'week']).join(weekly_flow_means, on=['year', 'week']).reset_index()
    
    weeks = weekly_flow_means.reset_index()

    X = weeks['week_index']
    y = weeks['flow_class']

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    week_idx_train, week_idx_test = list(sss.split(X, y))[0]

    weeks['split'] = weeks['week_index'].map(lambda x: 'train' if x in week_idx_train else 'test')
    df['split'] = df['week_index'].map(lambda x: 'train' if x in week_idx_train else 'test')
    
    return df[df['split'] == 'train'], df[df['split'] == 'test']

class FlowPhotoDataset(torch.utils.data.Dataset):
    def __init__(self, table, data_dir, col_filename='filename', col_label='flow_cfs', transform=None, label_transform=None):
        self.table = table
        self.data_dir = data_dir
        self.col_filename = col_filename
        self.col_label = col_label
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.table) # no. rows in csv file

    def __getitem__(self, idx):
        filename = self.table.iloc[idx][self.col_filename]
        img_path = os.path.join(self.data_dir, "images", filename)
        image = read_image(img_path) # read image file as tensor
        image = image / 255.0 # convert to float in [0,1]
        
        label = self.table.iloc[idx][self.col_label] # get label from table
        if self.transform:
            image = self.transform(image) # transform image
        if self.label_transform:
            label = self.label_transform(label) # transform label
        return image, label

def get_output_shape(model, input_shape=(1,3,224,224)):
    x = torch.randn(*input_shape)
    out = model(x)
    return out.shape

class ResNet18(nn.Module):
    """PyTorch ResNet-18 architecture.
    Attributes:
        pretrained (bool): whether to use weights from network trained on ImageNet
        truncate (int): how many layers to remove from the end of the network
    """
    def __init__(self, truncate=0):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        if truncate > 0:
            self.model = nn.Sequential(*list(self.model.children())[:-truncate])

        self.model.eval()

    def forward(self, x):
        x = self.model(x)
        return x

class FlowPhotoRegressionModel(nn.Module):
    def __init__(self, input_shape=(3, 384, 682), truncate=2, num_hlayers=[256, 64]):
        super(FlowPhotoRegressionModel, self).__init__()
        self.input_shape = input_shape
        self.input_nchannels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        
        self.resnetbody = ResNet18(truncate=truncate)
        num_filters = get_output_shape(self.resnetbody, input_shape=(1,*input_shape))[1]
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.fclayer_modules = [nn.Linear(num_filters, num_hlayers[0]), nn.ReLU()]
        for i in range(1, len(num_hlayers)):
            self.fclayer_modules.extend([nn.Linear(num_hlayers[i-1], num_hlayers[i]), nn.ReLU()])
        self.fclayer_modules.extend([nn.Linear(num_hlayers[-1], 1)])
        self.fclayers = nn.Sequential(*self.fclayer_modules)
        self.freeze_resnet()
    
    def freeze_resnet(self, freeze = True):
        for p in list(self.children())[0].parameters():
            p.requires_grad = not freeze
    
    def forward(self, x):
        x = self.resnetbody(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fclayers(x)
        x = x.squeeze()
        return x

def _get_train_data_loader(dataset, batch_size, **kwargs):
    logger.info(f"create train data loader: {batch_size}")
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            training_dir,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs
    )


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def train2(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_dataset, test_dataset = _load_dataset(args.data_dir)
    train_loader = _get_train_data_loader(train_dataset, is_distributed, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    model = Net().to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader, device)
    save_model(model, args.model_dir)


def test2(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(FlowPhotoRegressionModel())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    logger.info(f"save model: {path}")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

def _load_data_file(data_dir, filename):
    filepath = os.path.join(data_dir, filename)
    logger.info(f"load dataset: {filepath}")
    df = pd.read_csv(filepath, dtype={'flow_cfs': np.float32})
    # df = df.head(64*2)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(tz='US/Eastern')

    # filter by hour
    min_hour = 7
    max_hour = 18
    logger.info(f"filter(hour): {min_hour} to {max_hour}")
    df = df[df['timestamp'].dt.hour.between(min_hour, max_hour)]

    min_month = 4
    max_month = 11
    logger.info(f"filter(month): {min_month} to {max_month}")
    df = df[df['timestamp'].dt.month.between(min_month, max_month)]

    logger.info(f"dataset loaded\n  rows: {len(df)}\n  flow: {df.flow_cfs.mean():>.2f} cfs")
    return df
    
def _create_datasets(train_df, test_df, data_dir):
    logger.info(f"dataset split")
    logger.info(f"  train\n    rows: {len(train_df)}\n    flow: {train_df.flow_cfs.min():>.2f}, {train_df.flow_cfs.mean():>.2f}, {train_df.flow_cfs.max():>.2f} cfs")
    logger.info(f"  test\n    rows: {len(test_df)}\n    flow: {test_df.flow_cfs.min():>.2f}, {test_df.flow_cfs.mean():>.2f}, {test_df.flow_cfs.max():>.2f} cfs")
    
    img_dir = os.path.join(data_dir, "images")
    img_path = os.path.join(img_dir, train_df['filename'].iloc[0])
    print(f"loading first image: {img_path}")
    img = read_image(img_path)
    aspect = img.shape[2] / img.shape[1]
    transform = transforms.Compose([
        transforms.Resize([480,np.int32(480 * aspect)]),
        transforms.CenterCrop([384,np.int32(384 * aspect)])
    ])
    label_transform = transforms.Lambda(lambda y: np.log(y))

    train_dataset = FlowPhotoDataset(train_df, data_dir, transform=transform, label_transform=label_transform)
    test_dataset = FlowPhotoDataset(test_df, data_dir, transform=transform, label_transform=label_transform)
    
    return train_dataset, test_dataset
    
def train(dataloader, model, loss_fn, optimizer, device):
    num_batches = len(dataloader)
    total_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 1 == 0:
            print(f"train [batch {(batch + 1):>5d}/{num_batches:>5d}] loss: {loss.item():>7f}")
    avg_loss = total_loss / num_batches
    print(f"train [average] loss: {avg_loss:>8f}")

def test(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    total_loss = 0
    
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y).item()
            total_loss += loss
            if batch % 1 == 0:
                print(f"test [batch {(batch + 1):>5d}/{num_batches:>5d}] loss: {loss:>7f}")
    avg_loss = total_loss / num_batches
    print(f"test [average] loss): {avg_loss:>8f}\n")

def save_dataset(df, model_dir):
    df.to_csv(os.path.join(model_dir, "dataset.csv"), index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS", '[]')))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("SM_NUM_GPUS"))
    # parser.add_argument("--data-dir", type=str, default='.')
    parser.add_argument("--filename", type=str, default='images.csv')
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--unfreeze-after", type=int, default=2)

    args = parser.parse_args()
    
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    
    print("args:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    if (not os.path.exists(args.model_dir)):
        raise Exception(f"Model directory ({args.model_dir}) not found")
    
    df = _load_data_file(args.data_dir, args.filename)
    if (args.head is not None):
        df = df.head(args.head)
    
    train_df, test_df = split_weekly_flow(df, args.test_size, args.seed)
    train_dataset, test_dataset = _create_datasets(train_df, test_df, args.data_dir)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    img, label = train_dataset[0]
    print(f"shape: {tuple(img.shape)}")
    
    device = "cpu"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)
    
    model = FlowPhotoRegressionModel(input_shape=tuple(img.shape))
    model.to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    
    logger.info("train: start")
    epochs = args.epochs
    for epoch in range(epochs):
        print(f"epoch [{(epoch + 1):>2d}/{epochs:>2d}]")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)

        if (epoch+1) == args.unfreeze_after:
            print('\nmodel:  unfreeze resnet\n\n')
            model.freeze_resnet(False)

    save_model(model, args.model_dir)
    logger.info("train: end")
    
    logger.info("inference: start")
    model.eval()
    
    pred_df = pd.concat([train_df, test_df])
    pred_df.sort_values(by='timestamp', inplace=True, ignore_index=True)
    pred_dataset = FlowPhotoDataset(pred_df, args.data_dir, transform=test_dataset.transform, label_transform=test_dataset.label_transform)
    pred_dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    pred_values = []
    num_batches = len(pred_dataloader)
    with torch.no_grad():
        for batch, (X, y) in enumerate(pred_dataloader):
            logger.debug(f"  batch [{(batch + 1):>5d}/{num_batches:>5d}]")
            predictions = model(X.to(device)).cpu().numpy()
            for prediction in predictions:
                pred_values.append(prediction)
    pred_df['pred'] = pd.Series(pred_values)    
    save_dataset(pred_df, args.model_dir)
    logger.info("inference: end")
    # train(parser.parse_args())

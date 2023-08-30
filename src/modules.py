"""PyTorch modules for constructing DeepStreamflow models.

"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from utils import get_output_shape


class ResNet18(nn.Module):
    """PyTorch ResNet-18 architecture.
    Attributes:
        pretrained (bool): whether to use weights from network trained on ImageNet
        truncate (int): how many layers to remove from the end of the network
    """

    def __init__(self, pretrained=True, truncate=0):
        super(ResNet18, self).__init__()
        self.model = resnet18(pretrained=pretrained)
        if truncate > 0:
            self.model = nn.Sequential(*list(self.model.children())[:-truncate])

        self.model.eval()

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet50(nn.Module):
    """PyTorch ResNet-50 architecture.
    Attributes:
        pretrained (bool): whether to use weights from network trained on ImageNet
        truncate (int): how many layers to remove from the end of the network
    """

    def __init__(self, pretrained=True, truncate=0):
        super(ResNet50, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        if truncate > 0:
            self.model = nn.Sequential(*list(self.model.children())[:-truncate])

        self.model.eval()

    def forward(self, x):
        x = self.model(x)
        return x


class ResNetRegressionNet(nn.Module):
    """TODO:
    - Add docstrings
    """

    def __init__(
        self,
        input_shape=(3, 384, 512),
        transforms=[],
        resnet_size=50,
        truncate=2,
        pretrained=True,
        num_hlayers=[256, 64],
    ):
        super(ResNetRegressionNet, self).__init__()
        self.input_shape = input_shape
        self.input_nchannels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.transforms = transforms

        if resnet_size == 50:
            self.resnetbody = ResNet50(pretrained=pretrained, truncate=truncate)
        elif resnet_size == 18:
            self.resnetbody = ResNet18(pretrained=pretrained, truncate=truncate)
        else:
            raise ValueError("Only resnet_size 50 or 18 are supported.")
        num_filters = get_output_shape(self.resnetbody, input_shape=(1, *input_shape))[
            1
        ]
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fclayer_modules = [nn.Linear(num_filters, num_hlayers[0]), nn.ReLU()]
        for i in range(1, len(num_hlayers)):
            self.fclayer_modules.extend(
                [nn.Linear(num_hlayers[i - 1], num_hlayers[i]), nn.ReLU()]
            )
        self.fclayer_modules.extend([nn.Linear(num_hlayers[-1], 1)])
        self.fclayers = nn.Sequential(*self.fclayer_modules)

    def forward(self, x):
        x = self.resnetbody(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fclayers(x)
        x = x.squeeze()
        return x


class ResNetRankNet(ResNetRegressionNet):
    """TODO:
    - Add docstrings
    """

    def __init__(
        self,
        input_shape=(3, 384, 512),
        transforms=[],
        resnet_size=50,
        truncate=2,
        pretrained=True,
        num_hlayers=[256, 64],
    ):
        super().__init__(
            input_shape=input_shape,
            transforms=transforms,
            resnet_size=resnet_size,
            truncate=truncate,
            pretrained=pretrained,
            num_hlayers=num_hlayers,
        )

    def forward_single(self, x):
        x = self.resnetbody(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fclayers(x)
        x = x.squeeze()
        return x

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_single(input1)
        # forward pass of input 2
        output2 = self.forward_single(input2)
        # return the predicted feature vectors of both inputs
        return output1, output2

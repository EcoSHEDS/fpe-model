"""PyTorch modules for constructing DeepStreamflow models.

This module contains neural network architectures used for flow prediction and ranking,
built on top of ResNet backbones.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from utils import get_output_shape


class ResNet18(nn.Module):
    """PyTorch ResNet-18 architecture wrapper.
    
    A wrapper around torchvision's ResNet-18 that allows truncating layers from the end
    of the network. Uses pretrained weights from torchvision.
    
    Attributes:
        model (nn.Module): The underlying ResNet-18 model
    """

    def __init__(self, truncate=0):
        """Initialize the ResNet-18 model.
        
        Args:
            truncate (int): Number of layers to remove from the end of the network.
                          Default is 0 (use full network).
        """
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        if truncate > 0:
            self.model = nn.Sequential(*list(self.model.children())[:-truncate])

        self.model.eval()

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output features
        """
        return self.model(x)


class ResNet50(nn.Module):
    """PyTorch ResNet-50 architecture wrapper.
    
    A wrapper around torchvision's ResNet-50 that allows truncating layers from the end
    of the network. Uses pretrained weights from torchvision.
    
    Attributes:
        model (nn.Module): The underlying ResNet-50 model
    """

    def __init__(self, truncate=0):
        """Initialize the ResNet-50 model.
        
        Args:
            truncate (int): Number of layers to remove from the end of the network.
                          Default is 0 (use full network).
        """
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        if truncate > 0:
            self.model = nn.Sequential(*list(self.model.children())[:-truncate])

        self.model.eval()

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output features
        """
        return self.model(x)


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
        transforms=[],
        resnet_size=50,
        truncate=2,
        num_hlayers=[256, 64],
    ):
        """Initialize the regression network.
        
        Args:
            input_shape (tuple): Expected input shape (channels, height, width).
                               Default is (3, 384, 512).
            transforms (list): List of transforms to apply to inputs. Default is empty.
            resnet_size (int): Size of ResNet backbone (18 or 50). Default is 50.
            truncate (int): Number of layers to remove from ResNet. Default is 2.
            num_hlayers (list): List of hidden layer sizes for the fully connected
                              layers. Default is [256, 64].
        """
        super(ResNetRegressionNet, self).__init__()
        self.input_shape = input_shape
        self.input_nchannels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.transforms = transforms

        # Initialize ResNet backbone
        if resnet_size == 50:
            self.resnetbody = ResNet50(truncate=truncate)
        elif resnet_size == 18:
            self.resnetbody = ResNet18(truncate=truncate)
        else:
            raise ValueError("Only resnet_size 50 or 18 are supported.")
            
        # Get number of features from ResNet
        num_filters = get_output_shape(self.resnetbody, input_shape=(1, *input_shape))[1]
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Build fully connected layers
        self.fclayer_modules = [nn.Linear(num_filters, num_hlayers[0]), nn.ReLU()]
        for i in range(1, len(num_hlayers)):
            self.fclayer_modules.extend(
                [nn.Linear(num_hlayers[i - 1], num_hlayers[i]), nn.ReLU()]
            )
        self.fclayer_modules.extend([nn.Linear(num_hlayers[-1], 1)])
        self.fclayers = nn.Sequential(*self.fclayer_modules)

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Predicted scalar value for each input in the batch
        """
        x = self.resnetbody(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fclayers(x)
        return x.squeeze()


class ResNetRankNet(ResNetRegressionNet):
    """ResNet-based network for learning to rank images.
    
    Extends ResNetRegressionNet to handle pairs of images for learning to rank.
    The network produces scalar scores for each image that can be compared to
    determine relative ranking.
    
    Inherits all attributes from ResNetRegressionNet.
    """

    def __init__(
        self,
        input_shape=(3, 384, 512),
        transforms=[],
        resnet_size=50,
        truncate=2,
        num_hlayers=[256, 64],
    ):
        """Initialize the ranking network.
        
        Args: Same as ResNetRegressionNet
        """
        super().__init__(
            input_shape=input_shape,
            transforms=transforms,
            resnet_size=resnet_size,
            truncate=truncate,
            num_hlayers=num_hlayers,
        )

    def forward_single(self, x):
        """Forward pass for a single input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Predicted scalar score for ranking
        """
        x = self.resnetbody(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fclayers(x)
        return x.squeeze()

    def forward(self, input1, input2):
        """Forward pass for a pair of inputs.
        
        Args:
            input1 (torch.Tensor): First input tensor
            input2 (torch.Tensor): Second input tensor
            
        Returns:
            tuple: Predicted scalar scores for both inputs
        """
        output1 = self.forward_single(input1)
        output2 = self.forward_single(input2)
        return output1, output2


class ResNetRankNetWithAuxiliary(ResNetRankNet):
    """ResNet-based ranking network that combines image features with auxiliary data.
    
    Extends ResNetRankNet to handle both image data and additional numerical features
    (auxiliary data). The network concatenates the image features with auxiliary features
    before passing through the fully connected layers.
    
    Attributes:
        auxiliary_size (int): Number of additional numerical features
        auxiliary_encoder (nn.Module): Neural network to process auxiliary features
        All other attributes inherited from ResNetRankNet
    """

    def __init__(
        self,
        input_shape=(3, 384, 512),
        transforms=[],
        resnet_size=50,
        truncate=2,
        num_hlayers=[256, 64],
        auxiliary_size=1,
        auxiliary_encoding_size=32
    ):
        """Initialize the ranking network with auxiliary data support.
        
        Args:
            input_shape (tuple): Expected input shape (channels, height, width).
                               Default is (3, 384, 512).
            transforms (list): List of transforms to apply to inputs. Default is empty.
            resnet_size (int): Size of ResNet backbone (18 or 50). Default is 50.
            truncate (int): Number of layers to remove from ResNet. Default is 2.
            num_hlayers (list): List of hidden layer sizes for the fully connected
                              layers. Default is [256, 64].
            auxiliary_size (int): Number of auxiliary features (e.g., temperature, 
                                humidity, etc.). Default is 1.
            auxiliary_encoding_size (int): Size of the encoded auxiliary features before
                                         concatenation with image features. Default is 32.
        """
        # Initialize parent class without final FC layers
        super().__init__(
            input_shape=input_shape,
            transforms=transforms,
            resnet_size=resnet_size,
            truncate=truncate,
            num_hlayers=num_hlayers,
        )
        
        self.auxiliary_size = auxiliary_size
        
        # Create auxiliary encoder with normalization
        self.auxiliary_encoder = nn.Sequential(
            nn.BatchNorm1d(auxiliary_size),  # Add normalization
            nn.Linear(auxiliary_size, auxiliary_encoding_size),
            nn.ReLU(),
            nn.BatchNorm1d(auxiliary_encoding_size),  # Add normalization
            nn.Linear(auxiliary_encoding_size, auxiliary_encoding_size),
            nn.ReLU(),
            nn.BatchNorm1d(auxiliary_encoding_size)  # Add final normalization
        )
        
        # Get number of features from ResNet
        num_image_features = get_output_shape(self.resnetbody, 
                                            input_shape=(1, *input_shape))[1]
        
        # Rebuild FC layers with combined features
        combined_features = num_image_features + auxiliary_encoding_size
        self.fclayer_modules = [nn.Linear(combined_features, num_hlayers[0]), nn.ReLU()]
        for i in range(1, len(num_hlayers)):
            self.fclayer_modules.extend(
                [nn.Linear(num_hlayers[i - 1], num_hlayers[i]), nn.ReLU()]
            )
        self.fclayer_modules.extend([nn.Linear(num_hlayers[-1], 1)])
        self.fclayers = nn.Sequential(*self.fclayer_modules)

    def forward_single(self, x, auxiliary):
        """Forward pass for a single input with auxiliary data.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width)
            auxiliary (torch.Tensor): Auxiliary tensor of shape (batch_size, auxiliary_size)
            
        Returns:
            torch.Tensor: Predicted scalar score for ranking
        """
        # Process image through ResNet
        x = self.resnetbody(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten image features
        
        # Process auxiliary data
        auxiliary_features = self.auxiliary_encoder(auxiliary)
        
        # Concatenate image features with auxiliary features
        combined_features = torch.cat([x, auxiliary_features], dim=1)
        
        # Pass through FC layers
        x = self.fclayers(combined_features)
        return x.squeeze()

    def forward(self, input1, input2, auxiliary1, auxiliary2):
        """Forward pass for a pair of inputs with auxiliary data.
        
        Args:
            input1 (torch.Tensor): First input image tensor
            input2 (torch.Tensor): Second input image tensor
            auxiliary1 (torch.Tensor): First input auxiliary tensor
            auxiliary2 (torch.Tensor): Second input auxiliary tensor
            
        Returns:
            tuple: Predicted scalar scores for both inputs
        """
        output1 = self.forward_single(input1, auxiliary1)
        output2 = self.forward_single(input2, auxiliary2)
        return output1, output2

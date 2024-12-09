"""PyTorch modules for constructing DeepStreamflow models.

This module contains neural network architectures used for flow prediction and ranking,
built on top of ResNet backbones.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights
)
from utils import get_output_shape


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
        transforms=[],
        resnet_size=50,
        truncate=2,
        num_hlayers=[256, 64],
        dropout_rate=0.0,
        use_batch_norm=False,
    ):
        """Initialize the regression network.
        
        Args:
            input_shape (tuple): Expected input shape (channels, height, width).
                               Default is (3, 384, 512).
            transforms (list): List of transforms to apply to inputs. Default is empty.
            resnet_size (int): Size of ResNet backbone (18, 34, 50, 101, or 152). Default is 50.
            truncate (int): Number of layers to remove from ResNet. Default is 2.
            num_hlayers (list): List of hidden layer sizes for the fully connected
                              layers. Default is [256, 64].
            dropout_rate (float): Dropout rate for the fully connected layers. Default is 0.0.
            use_batch_norm (bool): Whether to use batch normalization. Default is False.
        """
        super(ResNetRegressionNet, self).__init__()
        self.input_shape = input_shape
        self.input_nchannels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.transforms = transforms

        # Initialize ResNet backbone
        self.resnetbody = ResNet(size=resnet_size, truncate=truncate)
            
        # Get number of features from ResNet
        num_filters = get_output_shape(self.resnetbody, input_shape=(1, *input_shape))[1]
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Build fully connected layers with optional dropout and batch norm
        self.fclayer_modules = []
        for i, (in_features, out_features) in enumerate(zip([num_filters] + num_hlayers[:-1], num_hlayers)):
            self.fclayer_modules.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features) if use_batch_norm else nn.Identity(),
                nn.ReLU()
            ])
            if dropout_rate > 0:
                self.fclayer_modules.append(nn.Dropout(dropout_rate))
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

    def get_features(self, x):
        """Extract features before the final layer.
        
        Useful for transfer learning or feature analysis.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Features from the penultimate layer
        """
        x = self.resnetbody(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Run through all layers except the last
        for layer in self.fclayer_modules[:-1]:
            x = layer(x)
        return x

    def initialize_weights(self, method='kaiming'):
        """Initialize network weights.
        
        Args:
            method (str): Initialization method ('none', 'kaiming', 'xavier', or 'normal')
        """
        if method == 'none':
            return  # Skip initialization
            
        for m in self.fclayer_modules:
            if isinstance(m, nn.Linear):
                if method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                elif method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'normal':
                    nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_model_summary(self):
        """Get a summary of the model architecture and parameters.
        
        Returns:
            dict: Model information including parameter counts and layer sizes
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'resnet_size': self.resnetbody.model.__class__.__name__,
            'fc_layers': [m.out_features for m in self.fclayer_modules if isinstance(m, nn.Linear)],
            'input_shape': self.input_shape
        }


class ResNetRegressionNetWithAuxiliary(ResNetRegressionNet):
    """ResNet-based regression network that combines image features with auxiliary data.
    
    Extends ResNetRegressionNet to handle both image data and additional numerical features
    (auxiliary data). The network concatenates the image features with auxiliary features
    before passing through the fully connected layers.
    
    Attributes:
        auxiliary_size (int): Number of additional numerical features
        auxiliary_encoder (nn.Module): Neural network to process auxiliary features
        All other attributes inherited from ResNetRegressionNet
    """

    def __init__(
        self,
        input_shape=(3, 384, 512),
        transforms=[],
        resnet_size=50,
        truncate=2,
        num_hlayers=[256, 64],
        auxiliary_size=1,
        auxiliary_encoding_size=32,
        dropout_rate=0.0,
        use_batch_norm=False,
    ):
        """Initialize the regression network with auxiliary data support.
        
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
            dropout_rate (float): Dropout rate for the fully connected layers. Default is 0.0.
            use_batch_norm (bool): Whether to use batch normalization. Default is False.
        """
        # Initialize parent class without final FC layers
        super().__init__(
            input_shape=input_shape,
            transforms=transforms,
            resnet_size=resnet_size,
            truncate=truncate,
            num_hlayers=num_hlayers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )
        
        self.auxiliary_size = auxiliary_size
        
        # Create auxiliary encoder (small neural network)
        self.auxiliary_encoder = nn.Sequential(
            nn.Linear(auxiliary_size, auxiliary_encoding_size),
            nn.ReLU(),
            nn.Linear(auxiliary_encoding_size, auxiliary_encoding_size),
            nn.ReLU()
        )
        
        # Get number of features from ResNet
        num_image_features = get_output_shape(self.resnetbody, 
                                            input_shape=(1, *input_shape))[1]
        
        # Rebuild FC layers with combined features
        combined_features = num_image_features + auxiliary_encoding_size
        self.fclayer_modules = []
        self.fclayer_modules.extend([
            nn.Linear(combined_features, num_hlayers[0]),
            nn.ReLU()
        ])
        if dropout_rate > 0:
            self.fclayer_modules.append(nn.Dropout(dropout_rate))
            
        for i in range(1, len(num_hlayers)):
            self.fclayer_modules.extend([
                nn.Linear(num_hlayers[i - 1], num_hlayers[i]),
                nn.ReLU()
            ])
            if dropout_rate > 0:
                self.fclayer_modules.append(nn.Dropout(dropout_rate))
        self.fclayers = nn.Sequential(*self.fclayer_modules)

    def forward(self, x, auxiliary):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width)
            auxiliary (torch.Tensor): Auxiliary tensor of shape (batch_size, auxiliary_size)
            
        Returns:
            torch.Tensor: Predicted scalar value for each input in the batch
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
        dropout_rate=0.0,
        use_batch_norm=False,
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
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
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
        auxiliary_encoding_size=32,
        dropout_rate=0.0,
        use_batch_norm=False,
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
            dropout_rate (float): Dropout rate for the fully connected layers. Default is 0.0.
            use_batch_norm (bool): Whether to use batch normalization. Default is False.
        """
        # Initialize parent class without final FC layers
        super().__init__(
            input_shape=input_shape,
            transforms=transforms,
            resnet_size=resnet_size,
            truncate=truncate,
            num_hlayers=num_hlayers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )
        
        self.auxiliary_size = auxiliary_size
        
        # Create auxiliary encoder (small neural network)
        self.auxiliary_encoder = nn.Sequential(
            nn.Linear(auxiliary_size, auxiliary_encoding_size),
            nn.ReLU(),
            nn.Linear(auxiliary_encoding_size, auxiliary_encoding_size),
            nn.ReLU()
        )
        
        # Get number of features from ResNet
        num_image_features = get_output_shape(self.resnetbody, 
                                            input_shape=(1, *input_shape))[1]
        
        # Rebuild FC layers with combined features
        combined_features = num_image_features + auxiliary_encoding_size
        self.fclayer_modules = []
        self.fclayer_modules.extend([
            nn.Linear(combined_features, num_hlayers[0]),
            nn.ReLU()
        ])
        if dropout_rate > 0:
            self.fclayer_modules.append(nn.Dropout(dropout_rate))
            
        for i in range(1, len(num_hlayers)):
            self.fclayer_modules.extend([
                nn.Linear(num_hlayers[i - 1], num_hlayers[i]),
                nn.ReLU()
            ])
            if dropout_rate > 0:
                self.fclayer_modules.append(nn.Dropout(dropout_rate))
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

    def get_model_summary(self):
        """Get a summary of the model architecture and parameters.
        
        Returns:
            dict: Model information including parameter counts and layer sizes
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'resnet_size': self.resnetbody.model.__class__.__name__,
            'fc_layers': [m.out_features for m in self.fclayer_modules if isinstance(m, nn.Linear)],
            'input_shape': self.input_shape
        }

    def initialize_weights(self, method='kaiming'):
        """Initialize network weights.
        
        Args:
            method (str): Initialization method ('none', 'kaiming', 'xavier', or 'normal')
        """
        if method == 'none':
            return  # Skip initialization
            
        for m in self.fclayer_modules:
            if isinstance(m, nn.Linear):
                if method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                elif method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'normal':
                    nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

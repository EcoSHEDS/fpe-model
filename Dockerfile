# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-ec2
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-ec2

RUN pip install --no-cache-dir mlflow

# Pre-download ResNet18 model weights to avoid downloading at runtime
RUN python3 -c "import torch; import torchvision.models as models; models.resnet18(pretrained=True)"

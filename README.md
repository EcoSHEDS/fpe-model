# Flow Photo Explorer Deep Learning Model

An [EcoSHEDS](https://www.usgs.gov/apps/ecosheds/) Project

:bangbang: **WARNING**: this repo is under heavy development. Use at your own risk.

## Python Environment

Create conda environment

```sh
conda env create -f environment.yml
```

Activate conda environment and start Jupyter Lab

```sh
conda activate fpe-model
jupyter lab
```

Then navigate to http://localhost:8888

## Datasets

Each dataset (flow and images) should be saved to `data/<site name>`. Within each site folder, there should be an `images/` folder containing the flow photos in `JPG` format and a `CSV` file called `images.csv` containing columns `filename`, `timestamp`, and `flow_cfs`.

## Development Notebooks

The `dev` folder contains a number of notebooks used during model development.

## Train Model

The regression model code is contained within the `fpe-regression.py` file. A similar file for the ranking model will soon be developed.

The `fpe-regression.py` file contains code for training the FPE PyTorch regression model. This file is designed to be uploaded to sagemaker, which trains the model using data stored in an S3 bucket. However, it can also be run locally at the command line.

This workflow was developed based on the MNIST tutorial: https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-python-sdk/pytorch_mnist

An application of the model is provided in `regression-parkers_brook.ipynb`.

## License

See `LICENSE`
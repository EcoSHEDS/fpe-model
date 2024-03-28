# Flow Photo Explorer Deep Learning Model

An [EcoSHEDS](https://www.usgs.gov/apps/ecosheds/) Project

:bangbang: **WARNING**: this repo is under heavy development. Use at your own risk.

## Background

This repo contains the source code for a deep learning model designed to estimate streamflow (or other hydrologic metrics) using timelapse imagery. This model will ultimately be integrated in the [EcoSHEDS Flow Photo Explorer](https://www.usgs.gov/apps/ecosheds/fpe/).

Original development of this model was completed by Amrita Gupta and Tony Chang at [Conservation Science Partners Inc](https://www.csp-inc.org/) with funding from the U.S. Geological Survey and the National Geographic Society. See the following paper for more information:

> Gupta, A., Chang, T., Walker, J., and Letcher, B. (2022). *Towards Continuous Streamflow Monitoring with Time-Lapse Cameras and Deep Learning.* In ACM SIGCAS/SIGCHI Conference on Computing and Sustainable Societies (COMPASS) (COMPASS '22). Association for Computing Machinery, New York, NY, USA, 353–363. https://doi.org/10.1145/3530190.3534805

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

Quick start:

```sh
FPE_DIR=D:/fpe/datasets
FPE_STATION=29
FPE_VARIABLE=FLOW_CFS

cd r
Rscript rank-dataset.R -d "${FPE_DIR}" -s "${FPE_STATION}" -v "${FPE_VARIABLE}" -o
# check annotations-cumul.png for training cutoff (annotations-end)
Rscript rank-input.R -d "${FPE_DIR}" -s "${FPE_STATION}" -v "${FPE_VARIABLE}" -o --min-hour=7 --max-hour=18 --annotations-end "2023-08-31"
```

### Flow Photo Dataset

A flow photo dataset contains the images and annotations associated with a single monitoring station and for a single variable. If observed data (e.g., streamflow) are available for that variable, then the images file includes the observed value at each image timestamp. These observed values are estimated using linear interpolation of the raw observations.

Datasets are stored using the following directory schema

`<STATION.NAME>/<VARIABLE>/<DATASET_ID>/data`

The `<DATASET_ID>` is typically a date stamp (`YYYYMMDD`).

For example: `~/fpe/West Brook 0_01171100/FLOW_CFS/20240326/data`.

Each dataset contains:

- `annotations.csv`: annotations dataset
- `annotations.png`: plot of annotations
- `images.csv`: images dataset (with observed data if available)
- `images.png`: timeseries plot of observed values for each image
- `station.json`: station info from database

A dataset is generated using the `dataset.R` script. 

```sh
cd r
Rscript dataset.R <STATION_ID> <VARIABLE_ID> </path/to/datasets>
Rscript dataset.R 29 FLOW_CFS D:/fpe/datasets
```

### Model Input Dataset

A model input dataset contains the images and annotations for a specific execution of the model training and inference pipeline.

This dataset is generated from a flow photo dataset, which contains all images and annotations for a single station.

The images dataset is filtered based on two sets of parameters:

- MIN_HOUR, MAX_HOUR: minimum and maximum hours of the day (local time) (e.g., MIN_HOUR=7, MAX_HOUR=18 yields a dataset containing only images from 7:00AM to 6:59PM)
- MIN_MONTH, MAX_MONTH: minimum and maximum month (e.g., MIN_MONTH=4, MAX_MONTH=11 yields dataset containing only images from Apr 1 - Nov 31)

The annotations dataset is filtered to only include image pairs where both images are in the filtered dataset. The annotations can then be further filtered based on:

- ANNOTATION_MIN_DATE, ANNOTATION_MAX_DATE: minimum and maximum dates of the annotation pairs

The model inputs will be saved using the following schema: 

`/path/to/<STATION.NAME>/<VARIABLE>/<DATASET_ID>/models/<MODEL_TYPE>/<MODEL_ID>`

Similar to the `<DATASET_ID>`, the `<MODEL_ID>` is typically a date stamp (`YYYYMMDD`), but does not necessarily need to match the `<DATASET_ID>` since the multiple models can be trained from the same dataset.

For example: `~/fpe/West Brook 0_01171100/FLOW_CFS/20240326/models/RANK/20240328`.

The inputs are generated using the `rank-input.R` script:

```sh
cd r
Rscript rank-input.R <DATASET/DIR> <STATION_ID> <VARIABLE_ID> <DATASET_ID> <MODEL_ID>
Rscript rank-input.R </path/to/datasets> 29 FLOW_CFS 20240326 20240328
```

## Development Notebooks

The `dev` folder contains a number of notebooks used during model development.

## Train Model

The regression model code is contained within the `fpe-regression.py` file. A similar file for the ranking model will soon be developed.

The `fpe-regression.py` file contains code for training the FPE PyTorch regression model. This file is designed to be uploaded to sagemaker, which trains the model using data stored in an S3 bucket. However, it can also be run locally at the command line.

This workflow was developed based on the MNIST tutorial: https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-python-sdk/pytorch_mnist

An application of the model is provided in `regression-parkers_brook.ipynb`.

## License

See `LICENSE`
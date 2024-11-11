# Flow Photo RankNet Model

An [EcoSHEDS](https://www.usgs.gov/apps/ecosheds/) Project

:bangbang: **WARNING**: this repo is under heavy development. Use at your own risk.

## Background

This repo contains the source code for a deep learning model designed to estimate streamflow (or other hydrologic metrics) using timelapse imagery. This model is integrated into the [EcoSHEDS Flow Photo Explorer](https://www.usgs.gov/apps/ecosheds/fpe/).

Original development of this model was completed by Amrita Gupta at [Microsoft AI For Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/) and Tony Chang at [Conservation Science Partners](https://www.csp-inc.org/) with funding from the U.S. Geological Survey and the National Geographic Society.

Preliminary research and model development can be found in:

> Gupta, A., Chang, T., Walker, J., and Letcher, B. (2022). *Towards Continuous Streamflow Monitoring with Time-Lapse Cameras and Deep Learning.* In ACM SIGCAS/SIGCHI Conference on Computing and Sustainable Societies (COMPASS) (COMPASS '22). Association for Computing Machinery, New York, NY, USA, 353–363. https://doi.org/10.1145/3530190.3534805

## Python Environment

Environment should match the python and torch versions of an [available SageMaker container](https://github.com/aws/deep-learning-containers/blob/master/available_images.md).

The FPE RankNet model currently uses python=3.9, torch=1.13.1, and torchvision=0.14.1.

If using Ubuntu, create conda environment:

```sh
conda env create -f environment.yml
```

Otherwise, create the environment manually:

```sh
conda create -n fpe-rank python=3.9
conda activate fpe-rank
conda config --env --add channels conda-forge
conda install jupyterlab numpy pandas scikit-learn tqdm
conda install boto3 sagemaker

# see https://pytorch.org/get-started/previous-versions/
pip install torch==1.13.1
pip install torchvision==0.14.1
# or
conda install pytorch torchvision torchaudio -c pytorch -c nvidia
# w/ cuda
conda install pytorch= torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Activate conda environment and start Jupyter Lab

```sh
conda activate fpe-rank
jupyter lab
```

Then navigate to http://localhost:8888

## Docker Container

Alternatively, you can run this code using one of the [AWS deep learning containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md), which allows you to replicate the SageMaker environment. See [instructions](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2-setup.html).

```sh
# first login to docker with aws creds
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
# pull image 
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-cpu-py310-ubuntu20.04-ec2
# run image
docker run -it 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-cpu-py310-ubuntu20.04-ec2
```

If a GPU is available, set up the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then add `--runtime=nvidia --gpus all` options to the docker command.

```sh
# fetch GPU container
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-ec2
docker run -it --runtime=nvidia --gpus all 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-ec2
nvidia-smi # run within container, should output GPU info
```

Test pytorch

```sh
mkdir /app
cd /app
git clone https://github.com/pytorch/examples.git
python examples/mnist/main.py
```

Run FPE training

```sh
docker run -it \
    --runtime=nvidia \
    --gpus all \
    -v /home/jeff/data/fpe/WESTB0/models/RANK-FLOW-20240424/input:/opt/ml/input/data/values \
    -v /home/jeff/data/fpe/WESTB0/images:/opt/ml/input/data/images \
    -v /opt/ml:/opt/ml \
    -v $(pwd):/app \
    -w /app \
    --env-file .env.docker \
    --network host \
    --shm-size=4g \
    763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-ec2

# interactively
python src/train.py

# check stats in separate terminal
docker ps # get container id
docker stats <container-id>
docker exec -it <container-id> /bin/bash
top
nvidia-smi
nvidia-smi dmon
```

## Datasets

Quick start:

```sh
FPE_DIR=D:/fpe/datasets
FPE_STATION=29
FPE_VARIABLE=FLOW_CFS

cd r
Rscript rank-dataset.R -d "${FPE_DIR}" -s "${FPE_STATION}" -v "${FPE_VARIABLE}" -o
# check annotations-cumul.png for training cutoff (annotations-end)
Rscript rank-input.R -d "${FPE_DIR}" -s "${FPE_STATION}" -v "${FPE_VARIABLE}" -o
```

### Flow Photo Dataset

A flow photo dataset contains the images and annotations associated with a single monitoring station and for a single variable. If observed data (e.g., streamflow) are available for that variable, then the images file includes the observed value at each image timestamp. These observed values are estimated using linear interpolation of the raw observations.

Datasets are stored using the following directory schema

`<STATION.NAME>/<VARIABLE>/<DATASET_VERSION>/data`

The `<DATASET_VERSION>` is typically a date stamp (`YYYYMMDD`).

For example: `~/fpe/West Brook 0_01171100/FLOW_CFS/20240326/data`.

Each dataset contains:

- `pairs.csv`: annotations dataset
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

- `ANNOTATION_MIN_DATE, ANNOTATION_MAX_DATE`: minimum and maximum dates of the annotation pairs

The model inputs will be saved using the following schema:

`/path/to/<STATION.NAME>/<VARIABLE>/<DATASET_VERSION>/models/<MODEL_VERSION>`

Similar to the `<DATASET_VERSION>`, the `<MODEL_VERSION>` is typically a date stamp (`YYYYMMDD`), but does not necessarily need to match the `<DATASET_VERSION>` since the multiple models can be trained from the same dataset.

For example: `~/fpe/rank/West Brook 0_01171100/FLOW_CFS/20240326/models/20240328`.

The inputs are generated using the `rank-input.R` script:

```sh
cd r
Rscript rank-input.R <DATASET/DIR> <STATION_ID> <VARIABLE_ID> <DATASET_VERSION> <MODEL_VERSION>
Rscript rank-input.R </path/to/datasets> 29 FLOW_CFS 20240326 20240328
```

## Model Training

```sh
python src/run-train.py --station-id 68 --directory=/mnt/d/fpe/rank --model-code RANK-FLOW-20240410
```

## Model Inference

```sh
# run training job
python src/run-transform.py --station-id 68 --directory=/mnt/d/fpe/rank --model-code RANK-FLOW-20240410
# wait for transform job to complete
python src/run-transform-merge.py --station-id 68 --directory=/mnt/d/fpe/rank --model-code RANK-FLOW-20240410
# wait for merge to complete (fpe-prod-lambda-models)
python src/run-transform-predictions.py --station-id 68 --directory=/mnt/d/fpe/rank --model-code RANK-FLOW-20240410
```

## Deploy

Copy diagnostics report and predictions CSV to S3.

```sh
./batch-deploy.sh /mnt/d/fpe/rank/stations-wb-model-uuid.txt /mnt/d/fpe/rank RANK-FLOW-20240410
```

## License

See `LICENSE`
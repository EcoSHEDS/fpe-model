# Instructions for Ensemble Team

This project involves the following steps:

1. [Set up AWS Infrastructure](#aws-infrastructure)
2. [Set up Development Environment](#development-environment)
3. [Import Datasets and Images](#import-datasets-and-images)
4. [Run Manual Model Pipeline](#run-manual-model-pipeline)
5. [Create Automated Pipeline](#develop-automated-pipeline)

Links:

- [Flow Photo Explorer | USGS](https://www.usgs.gov/apps/ecosheds/fpe/#/)
- [EcoSHEDS/fpe](https://github.com/ecosheds/fpe): source code for web app, database, API, and AWS infrastructure
- [EcoSHEDS/fpe-model](https://github.com/ecosheds/fpe-model): source code for ranking model and associated data processing scripts

## AWS Infrastructure

The AWS infrastructure is defined by CloudFormation templates in [`fpe/cloudformation/templates`](https://github.com/EcoSHEDS/fpe/tree/main/cloudformation/templates). However, only a few of these resources are needed for this project:

- S3 Buckets
	- `s3-storage` ([template](https://github.com/EcoSHEDS/fpe/blob/main/cloudformation/templates/s3-storage.json)): stores the images, user-uploaded datasets, user-generated annotation files, and model outputs that are publicly available (i.e., the files loaded by the web app)
	- `s3-models` ([template](https://github.com/EcoSHEDS/fpe/blob/main/cloudformation/templates/s3-model.json)): stores model inputs/outputs, not publicly accessible
- Lambda Functions
	- `lambda-models` ([template](https://github.com/EcoSHEDS/fpe/blob/main/cloudformation/templates/lambda-models.json), [source code](https://github.com/EcoSHEDS/fpe/tree/main/lambda/models)): a simple function to combine batch inference output files. SageMaker Batch Inference generates one file for each image resulting in thousands of small files. This function reads and merges these files into a single CSV file.
- RDS Instance ([template](https://github.com/EcoSHEDS/fpe/blob/main/cloudformation/templates/db.json))
	- `db-fpe` ([source code](https://github.com/EcoSHEDS/fpe/tree/main/db)): PostgreSQL database storing stations, image metadata, model metadata, etc. Database schema managed using [knex.js migrations](https://knexjs.org/guide/migrations.html).
- SageMaker Permissions ([template](https://github.com/EcoSHEDS/fpe/blob/main/cloudformation/templates/sagemaker.json)): IAM roles for SageMaker to access S3 buckets and run training and batch inference jobs.

## Development Environment

First, clone the main FPE repo (`EcoSHEDS/fpe`) and install npm dependencies. FPE currently runs on Node.js v18, but newer versions will probably work fine.

```sh
git clone https://github.com/EcoSHEDS/fpe.git
cd fpe
npm install
```

Set up `.env` configuration file for `development` environment to connect to database and access S3 storage bucket. These settings are loaded by `knexfile.js`.

```sh
# .env.development.local

# database creds
DB_HOST=<hostname>
DB_PORT=5432
DB_DATABASE=<dbname>
DB_USER=<username>
DB_PASSWORD=<password>

# name of `storage` bucket
BUCKET=my-fpe-s3-storage
```

Then clone the `fpe-model` repo in a separate folder.

```sh
cd .. # back out of fpe/
git clone https://github.com/EcoSHEDS/fpe-model.git
```

## Import Datasets and Images

Importing a development dataset involves two steps:

1. Importing data into the PostgreSQL database (`db-fpe`) using [knex migrations](https://knexjs.org/guide/migrations.html) and the [knex seed utility](https://knexjs.org/guide/migrations.html#seed-cli).

```sh
# download and extract fpe dataset to db/seeds/development
cd ~/fpe/db/seeds/development/data/users/d72c3799-4ca0-48cf-9d43-145a95d45bd9/stations
wget https://example.com/WESTB0-20220201-20230131-20240710.tar.gz # real URL will be emailed to you
tar -xzf WESTB0-20220201-20230131-20240710.tar.gz # should create WESTB0-20220201-20230131/ folder containing db/ and storage/ subfolders
rm WESTB0-20220201-20230131-20240710.tar.gz # delete after extraction

# create directory for imagesets
cd WESTB0-20220201-20230131
mkdir -p storage/imagesets/
cd storage/imagesets/

# download imageset tarballs from s3
for UUID in $(ls ../../db/imagesets/); do wget https://usgs-chs-conte-prod-fpe-storage.s3.amazonaws.com/seeds/imagesets/${UUID}.tar.gz; done

# extract tarballs
for FILE in *.tar.gz; do tar -xzvf $FILE; done

# delete tarballs
rm *.tar.gz

# check that the dataset and images are correctly extracted into this structure:
# ./fpe/db/seeds/development/data/
# ├── modelTypes.json
# ├── users
# │   └── d72c3799-4ca0-48cf-9d43-145a95d45bd9
# │       ├── stations
# │       │   └── WESTB0-20220201-20230131
# │       │       ├── db
# │       │       │   ├── annotations.json
# │       │       │   ├── imagesets
# │       │       │   │   ├── 265292ae-007e-4a94-a86c-e01028d85c1f
# │       │       │   │   │   ├── images.json
# │       │       │   │   │   └── imageset.json
# │       │       │   │   └── ...
# │       │       │   └── station.json
# │       │       └── storage
# │       │           ├── annotations
# │       │           │   ├── 0406f678-d037-4090-8c3e-411d33c258b4.json
# │       │           │   └── ...
# │       │           └── imagesets
# │       │               ├── 265292ae-007e-4a94-a86c-e01028d85c1f
# │       │               │   ├── images
# │       │               │   │   ├── West Brook 0__2022-12-20__09-54-55(1).JPG
# │       │               │   │   └── ...
# │       │               │   ├── pii.json
# │       │               │   └── thumbs
# │       │               │       ├── West Brook 0__2022-12-20__09-54-55(1).JPG
# │       │               │       └── ...
# │       │               └── ...
# │       └── user.json
# └── variables.json

# go back to ./db folder in fpe repo
cd ~/fpe/db

# set environment so creds are loaded from .env.development.local
export NODE_ENV=development

# run migrations to set up db schema
knex migrate:latest

# run seeds to import data from ./development/data
knex seed:run
```

2. Uploading image files to the S3 storage bucket (`s3-storage`) using `aws` CLI.

```sh
cd ~/fpe/db/seeds/development/data/users/d72c3799-4ca0-48cf-9d43-145a95d45bd9/stations

# upload everything in the storage/ folder to the s3 bucket
aws sync WESTB0-20220201-20230131/storage/ s3://my-fpe-s3-storage/ # uploads imagesets/ and annotations/ to s3-storage

# note: in s3, the data are *not* grouped by station
# so imagesets should be saved to s3://my-fpe-s3-storage/imagesets/<uuid>
# and annotations should be saved to s3://my-fpe-s3-storage/annotations/<uuid>.json
```

At this point, the PostgreSQL database should be populated with data, and the associated images available in the S3 storage bucket.

## Run Manual Model Pipeline

With the datasets and images now available in AWS, run the following scripts to:

1. Create images and annotation dataset for a given station
2. Create model training dataset
3. Run model training using annotations
4. Run model batch inference on all images
5. Post-process batch inference results
6. Generate model diagnostics report
7. Upload model results to database and S3

### 1. Create Station Images and Annotations Datasets

First, create a local folder for storing the model files, for example `~/data/fpe`. This will serve as the root directory for generating datasets and model inputs.

Next, create the full image and annotation datasets for a given station using the `r/rank-dataset.R` script.

```sh
cd r
Rscript rank-dataset.R --help # list arguments
Rscript rank-dataset -d ~/data/fpe -s 29 -v FLOW_CFS -o RANK-FLOW-20240709
# -d is root directory for model files
# -s is the station ID (see stations table in database)
# -v is the variable code (see variables table in database, though right now we are only focused on FLOW_CFS)
# -o forces overwrite if dataset files already exist
# RANK-FLOW-20240709 is a positional argument defining the dataset "code", which is used for dataset versioning
```

This script will generate a new folder for the given station at `~/data/fpe/29 - West Brook Zero/FLOW_CFS/datasets/RANK-FLOW-20240709`. Within that folder, are a set of files, most of which are used for diagnostics. The two primary dataset files are:

- `images.csv`: a table of all images for the station along any available observed data (`value`) retrieved from FPE (user-uploaded) or USGS NWIS (if the station has an `nwis_id`)
- `annotations.csv`: a table of all pairwise annotations

### 2. Create Model Training Dataset

After creating the full images and annotations dataset, the next step is to generate a model input dataset for the station using the `r/rank-input.R` script.

```sh
Rscript rank-input.R --help # list arguments
Rscript rank-input.R -d ~/data/fpe -s 29 -v FLOW_CFS -D RANK-FLOW-20240709 RANK-FLOW-20240709
# -d is root directory for model files
# -s is the station ID
# -v is the variable code
# -D is the dataset code from the previous step
# RANK-FLOW-20240709 is a positional argument defining the *model* code, which is used for model versioning
```

This script loads the `images.csv` and `annotations.csv` files from the dataset, splits the annotations dataset into training/testing subsets, and then generates the input files that are used by SageMaker. The model input dataset will be saved to `~/data/fpe/29 - West Brook Zero/FLOW_CFS/models/RANK-FLOW-20240709`. The primary input files are:

- `pairs.csv`: annotations used for training with the `split` column assigning each pair to `train` or `test`
- `manifest.json`: lists the S3 bucket and keys for all images referenced in `pairs.csv` and thus needed for training (note this is only a subset of the full images dataset as not all images are annotated).
- `images.csv`: the full images dataset, which is used after training to perform batch inference

Note that the `rank-input.R` script accepts a number of additional parameters for filtering the dataset. For example, we currently exclude night-time photos, which are (roughly) identified as any between 7PM and 7AM. This window can be changed using the `MIN_HOUR` and `MAX_HOUR` arguments to `rank-input.R`, which default to `MIN_HOUR=7` and `MAX_HOUR=18`, respectively. Similarly, there are arguments for `MIN_MONTH` and `MAX_MONTH` to exclude winter time when snow and ice can be an issue. However, currently we include all months.

### 3. Run Model Training

With the input dataset created, the ranking model can now be trained using SageMaker.

```sh
cd .. # back to root of repo
python src/run-train.py --directory ~/data/fpe --station-id 29 --model-code RANK-FLOW-20240709
# --directory is root directory of model files
# --station-id is station ID
# --model-code is the model code
```

This script uploads the model input files to S3, creates an instance of `sagemaker.pytorch.PyTorch` (see [sdk](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html)) that refers to the model source code (`src/train.py`), location of input files in S3, and defines the training hyperparameters. The model is then trained using the `.fit()` method, which initiates a SageMaker training job that can be monitored in the AWS console.

### 4. Run Model Batch Inference

After the training job is complete, the trained model is used to generate predicted scores for every image at the given station using a SageMaker [Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html) job.

```sh
python src/run-transform.py --directory=~/data/fpe --station-id=29 --model-code=RANK-FLOW-20240709
```

This script creates a new `manifest.json` file listing the S3 locations of all images for the station, uploads it to S3, and then runs a batch inference job using the `PyTorchModel.transformer.transform()` method. This job can then be monitored in the AWS console.

### 5. Post-Process Batch Inference Output

The batch transform job generates one file for each image using the image key and appending `.out`. For example, the prediction for `path/to/IMAGE-0001.jpg` is `path/to/IMAGE-0001.jpg.out`. Two post-processing scripts are then used to combine these predictions into a single csv file.

The first script `run-transform-merge.py` uses the `lambda-model` Lambda function to read a set of batch inference output files, and merge them into a single CSV. Due to the lambda runtime limitations (max 15 minutes), this process is broken up into batches by invoking the lambda function multiple times, once for each subset of images. By default, the batch size is 5000 meaning this script will generate one CSV file for every 5000 images, which are then named `predictions-{start index}-{end index}.csv`. For example, if a station has 8,123 images total, then `run-transform-merge.py` should generate two CSV files: `predictions-00000-04999.csv` and `predictions-05000-09999.csv` (note: the latter file will only contain 3,123 rows).

```sh
python src/run-transform-merge.py --station-id 29 --directory=~/data/fpe --model-code RANK-FLOW-20240709
# note same arguments as run-transform.py
```

After the lambda function has completed all invocations, the last script merges the `predictions-XXX-YYY.csv` files into one final dataset `predictions.csv`.

```sh
python src/run-transform-predictions.py --station-id 29 --directory=~/data/fpe --model-code RANK-FLOW-20240709
# note same arguments as run-transform.py
```

The final `predictions.csv` file is saved to the model folder at `~/data/fpe/<station>/FLOW_CFS/models/<model code>/transform/predictions.csv`.
### 6. Generate Model Diagnostics Report

After the model training and batch transform are complete, the `r/prediction-report.R` script is used to generate a model diagnostics report using quarto. This script should be run interactively by calling the `generate_report()` function. The diagnostics report is saved in the model folder at `~/data/fpe/<station>/FLOW_CFS/models/<model code>/<model code>.html`.

### 7. Upload Model Results

Lastly, to make the model results available on the FPE website, the model metadata needs to be saved to the database, and the `predictions.csv` and diagnostics report uploaded to S3. This is a more manual process facilitated by a couple scripts.

```sh
cd r/
Rscript rank-model-db.R -t RANK -v FLOW_CFS -c RANK-FLOW-20240709 -s https://usgs-chs-conte-prod-fpe-storage.s3.us-west-2.amazonaws.com/models ~/data/fpe/stations.txt
```

The `stations.txt` file is a simple text file listing a set of station IDs so that the model files can be uploaded for multiple stations as a batch. To upload for just one station, this file can contain a single ID (e.g. `29`).

The `rank-model-db.R` generates two files in the same folder as `stations.txt`:

- `stations-models-db.csv` contains the model metadata. This file is manually imported into the `models` table in the Postgres database using an SQL client or `psql`.
- `stations-models-uuid.txt` contains a UUID assigned to each model. This UUID is used as a unique identifier for each model, and is used for uploading the model output files (`predictions.csv` and `{MODEL_CODE}.html`) to the S3 storage bucket at `models/{UUID}/`.

Lastly, use the `batch-deploy.sh` to upload the model files to S3 by specifying the file containing the UUID for each model, the model root directory, and the model code.

```sh
./batch-deploy.sh ~/data/fpe/stations-models-uuid.txt ~/data/fpe RANK-FLOW-20240709
```

## Develop Automated Pipeline

Our next objective is to automate the complete model pipeline described in the previous section using a [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html). The pipeline would be designed to train and apply a model for a single station. The inputs to this pipeline would include the following:

```sh
# model selection
model_type="RANK" # currently, we only have one model, but we may add more (e.g., a classification model for predicting flow/no flow)
variable_code="FLOW_CFS" # currently, we are focused only on flow, but plan to support different variables in the future.

# station
station_id=29 # refers to stations.id column in database

# input dataset parameters (see r/rank-input.R)
# these are just a start, we will be adding more
min_hour=7
max_hour=18
min_month=1
max_month=12

# training hyperparameters (see src/train.py)
# we'll be adding more to define the image transforms (e.g. decolorize, jitter, etc)
n_epochs=20
learning_rate=0.001
```

Given this set of parameters, the model pipeline would then generate the model input dataset, train the model, run batch inference, post-process the results, and save the model metadata to the database and output files to S3 so they are available on the FPE website.

The final pipeline will need to be fully scripted and defined using a [CloudFormation template](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-sagemaker-pipeline.html) so that it can be readily deploy to the USGS AWS account (which only allows for the creation of resources via cloudformation).

This work will likely involve changes to the existing scripts and code. Therefore, all work should be completed under a separate branch (e.g., `ensemble`) in the `fpe-model` repo.





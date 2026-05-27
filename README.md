# USGS Flow Photo Explorer Model

## Background

This repo contains the source code for a deep learning model designed to estimate streamflow (or other hydrologic metrics) using timelapse imagery. This model is integrated into the [EcoSHEDS Flow Photo Explorer](https://www.usgs.gov/apps/ecosheds/fpe/).

Initial development of this model was completed by Amrita Gupta at [Microsoft AI For Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/) and Tony Chang at [Conservation Science Partners](https://www.csp-inc.org/) with funding from the U.S. Geological Survey and the National Geographic Society.

Preliminary research and model development can be found in:

> Gupta, A., Chang, T., Walker, J., and Letcher, B. (2022). *Towards Continuous Streamflow Monitoring with Time-Lapse Cameras and Deep Learning.* In ACM SIGCAS/SIGCHI Conference on Computing and Sustainable Societies (COMPASS) (COMPASS '22). Association for Computing Machinery, New York, NY, USA, 353–363. https://doi.org/10.1145/3530190.3534805

The model was then operationalized for training and inference using AWS SageMaker by Jeff Walker at [Walker Environmental Research](https://walkerenvres.com).

Ongoing model development is continuing with contributions from Amrita Gupta [Microsoft AI For Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/) and Jeff Walker at [Walker Environmental Research](https://walkerenvres.com) as part of the USGS EcoSHEDS project.

## Python Environment

Environment should match the python and torch versions of an [available SageMaker container](https://github.com/aws/deep-learning-containers/blob/master/available_images.md).

The FPE RankNet model currently uses python=3.9, torch=1.13.1, and torchvision=0.14.1.

Install Python with pyenv:

```sh
pyenv install -s 3.9.25
pyenv local 3.9.25
```

Create and activate a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```sh
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-dev.txt
```

Start Jupyter Lab:

```sh
jupyter lab
```

Then navigate to http://localhost:8888

## Model Pipeline

These instructions assume the FPE database, R configuration, AWS profile, and S3 buckets already exist. The Python scripts use the `conte-prod` AWS profile in `us-west-2`; the R scripts read database settings through the R project configuration.

Use a working directory to hold generated datasets, model inputs, downloaded model artifacts, predictions, and reports. A station list is a plain text file with one station ID per line.

```sh
FPE_DIR=/mnt/d/fpe/rank
STATIONS_FILE="${FPE_DIR}/stations.txt"
STATION_ID=29
VARIABLE_ID=FLOW_CFS
DATASET_CODE=RANK-FLOW-20240709
MODEL_CODE=RANK-FLOW-20240709
```

The pipeline writes files using this layout:

```text
${FPE_DIR}/<station-id>/datasets/<dataset-code>/
${FPE_DIR}/<station-id>/models/<model-code>/
```

### 1. Create Station Datasets

Run the R dataset script from the `r/` directory. It reads station metadata, image metadata, observations, and annotations from the FPE database/S3, then writes `images.csv`, `annotations.csv`, `station.json`, summary JSON, and diagnostic plots.

```sh
cd r
Rscript rank-dataset.R \
  --directory "${FPE_DIR}" \
  --station-id "${STATION_ID}" \
  --variable-id "${VARIABLE_ID}" \
  --overwrite \
  "${DATASET_CODE}"
```

For a batch of stations:

```sh
cd r
./batch-rank-dataset.sh "${STATIONS_FILE}" "${FPE_DIR}" "${VARIABLE_ID}" "${DATASET_CODE}"
```

The dataset is saved to `${FPE_DIR}/${STATION_ID}/datasets/${DATASET_CODE}`. Check `annotations-cumul.png`, `images.png`, and `rank-dataset.json` before creating model inputs.

### 2. Create Model Inputs

Still from `r/`, create the SageMaker input files. This script filters images and annotations, creates train/validation pair splits, duplicates reversed pairs, and writes `input/images.csv`, `input/annotations.csv`, `input/pairs.csv`, `input/manifest.json`, `input/station.json`, `input/rank-input.json`, and plots under the model directory.

```sh
Rscript rank-input.R \
  --directory "${FPE_DIR}" \
  --station-id "${STATION_ID}" \
  --variable-id "${VARIABLE_ID}" \
  --dataset-code "${DATASET_CODE}" \
  --overwrite \
  "${MODEL_CODE}"
```

Useful filters include `--min-hour`, `--max-hour`, `--min-month`, `--max-month`, `--images-start`, `--images-end`, `--annotations-start`, `--annotations-end`, `--annotations-max-n`, and `--train-frac`. By default, the script keeps images from 7 AM through 6 PM local time and all months.

For a batch of stations:

```sh
cd r
./batch-rank-input.sh "${STATIONS_FILE}" "${FPE_DIR}" "${VARIABLE_ID}" "${DATASET_CODE}" "${MODEL_CODE}"
```

The model inputs are saved to `${FPE_DIR}/${STATION_ID}/models/${MODEL_CODE}/input`.

### 3. Train Models

Run SageMaker training from the repository root. `run-train.py` uploads the model input directory to the private model S3 bucket, starts a PyTorch 1.13.1 / Python 3.9 training job, and writes the SageMaker job name to `${FPE_DIR}/${STATION_ID}/models/${MODEL_CODE}/job.txt`.

```sh
cd ..
python src/run-train.py \
  --station-id "${STATION_ID}" \
  --directory "${FPE_DIR}" \
  --model-code "${MODEL_CODE}"
```

For a batch of stations:

```sh
./batch-run.sh train "${STATIONS_FILE}" "${FPE_DIR}" "${MODEL_CODE}"
```

The script starts training with `wait=False`; monitor jobs in SageMaker before continuing. To stop a training job:

```sh
python src/stop-train.py --job-name fpe-rank-YYYYMMDD-HHMMSS
```

### 4. Run Batch Inference

After training finishes, start SageMaker Batch Transform from the repository root. `run-transform.py` reads `job.txt`, points SageMaker at the trained `model.tar.gz`, uploads a transform manifest for every image in `input/images.csv`, and starts a batch transform job.

```sh
python src/run-transform.py \
  --station-id "${STATION_ID}" \
  --directory "${FPE_DIR}" \
  --model-code "${MODEL_CODE}"
```

For a batch of stations:

```sh
./batch-run.sh transform "${STATIONS_FILE}" "${FPE_DIR}" "${MODEL_CODE}"
```

The script starts transform jobs with `wait=False`; monitor jobs in SageMaker before merging outputs. To stop a transform job:

```sh
python src/stop-transform.py --job-name fpe-rank-transform-<station-id>-YYYYMMDD-HHMMSS
```

### 5. Merge Predictions

After Batch Transform finishes, invoke the model Lambda merger. `run-transform-merge.py` asks `fpe-prod-lambda-models` to combine per-image transform outputs into chunked prediction CSVs in S3. The default chunk size is 5000 images.

```sh
python src/run-transform-merge.py \
  --station-id "${STATION_ID}" \
  --directory "${FPE_DIR}" \
  --model-code "${MODEL_CODE}"
```

For a batch of stations:

```sh
./batch-run.sh transform-merge "${STATIONS_FILE}" "${FPE_DIR}" "${MODEL_CODE}"
```

Wait for the Lambda invocations to finish, then download and combine the chunked predictions. `run-transform-predictions.py` also downloads the SageMaker `output.tar.gz` and `model.tar.gz` into `${FPE_DIR}/${STATION_ID}/models/${MODEL_CODE}/output`.

```sh
python src/run-transform-predictions.py \
  --station-id "${STATION_ID}" \
  --directory "${FPE_DIR}" \
  --model-code "${MODEL_CODE}"
```

For a batch of stations:

```sh
./batch-run.sh transform-predictions "${STATIONS_FILE}" "${FPE_DIR}" "${MODEL_CODE}"
```

The final predictions file is saved to `${FPE_DIR}/${STATION_ID}/models/${MODEL_CODE}/transform/predictions.csv`.

### 6. Generate Diagnostics Reports

Generate reports from the `r/` directory after `output/metrics.csv`, `output/args.json`, and `transform/predictions.csv` exist. The report renderer is currently defined in `prediction-report.R` as `generate_report(station_id, model_code, directory)`, and the bottom of that file contains the current batch loop. Update the station file, model code, and directory in that batch block, then run the script from `r/`.

```sh
cd r
Rscript prediction-report.R
```

Each report is copied to `${FPE_DIR}/<station-id>/models/${MODEL_CODE}/${MODEL_CODE}.html`.

### 7. Create Database Rows And Deploy Files

Create model metadata rows from the station list. Run this from `r/`; it writes `<stations-file-stem>-models-db.csv` and `<stations-file-stem>-models-uuid.txt` next to the station list.

```sh
cd r
Rscript rank-model-db.R \
  --model-type RANK \
  --variable-id "${VARIABLE_ID}" \
  --model-code "${MODEL_CODE}" \
  --s3-url https://usgs-chs-conte-prod-fpe-storage.s3.us-west-2.amazonaws.com/models \
  "${STATIONS_FILE}"
```

Manually import the generated `*-models-db.csv` into the FPE database `models` table.

Then upload model files to the public storage bucket from the repository root. The deploy script uploads `input/annotations.csv`, `input/images.csv`, `output/model.tar.gz`, the diagnostics HTML report, and `transform/predictions.csv` for each station/UUID pair.

```sh
cd ..
./batch-deploy.sh "${FPE_DIR}/stations-models-uuid.txt" "${FPE_DIR}" "${MODEL_CODE}"
```

## License

See `LICENSE`

FPE Personal Identifying Information (PII) Detector
---------------------------------------------------

*Summary:* detect PII in FPE images using MegaDetector in AWS SageMaker

*References:*
- [microsoft/CameraTraps](https://github.com/microsoft/CameraTraps): MegaDetector model
- [ecologize/yolov5](https://github.com/ecologize/yolov5): ML framework used to develop megadetector (note: fork pinned to version compatible with MegaDetector)
- [rbavery/animal_detector](https://github.com/rbavery/animal_detector): streamlit app that uses MegaDetector to detect animals, contains code to deploy MegaDetector using TorchScript (`/mdv5app`)
- [aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve](https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve/blob/main/notebook/04_SageMaker.ipynb): notebook showing how to deploy torchserve model to SageMaker
- [AWS Sagemaker | Use Your Own Inference Code with Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-batch-code.html)
- [AWS ML Blog | Run computer vision inference on large videos with Amazon SageMaker asynchronous endpoints](https://aws.amazon.com/blogs/machine-learning/run-computer-vision-inference-on-large-videos-with-amazon-sagemaker-asynchronous-endpoints/): using asynchronous endpoints

## Instructions for Deploying to Sagemaker

### Python Environment

Set working directory to `pii-detector/`.

```sh
cd pii-detector
```

Set up conda environment:

```sh
conda create -n fpe-pii python=3.8
conda activate fpe-pii
pip install -r lib/yolov5/requirements.txt torch-model-archiver "dvc[s3]"
```

Set up dvc by adding credentials to access S3 bucket (`walkerenvres-fpe-models`)

```
dvc remote modify --local storage access_key_id ''
dvc remote modify --local storage secret_access_key ''
```

*Step 1*: Fetch megadetector checkpoints (.pt) files from github

```sh
./fetch-mdv5.sh # fetch from Github
dvc pull        # or pull via dvc
```

*Step 2*: Convert megadetector weights to model archive and compress

```sh
./build-mdv5.sh a
./build-mdv5.sh b
```

*Step 3*: Upload to S3 model bucket

```sh
aws s3 cp model/mdv5a.tar.gz s3://<MODEL_BUCKET>/<PREFIX>/
aws s3 cp model/mdv5b.tar.gz s3://<MODEL_BUCKET>/<PREFIX>/
```

### Run TorchServe Locally

Run torchserve via docker

```sh
./docker-mdv5.sh
# example request
# curl http://127.0.0.1:8080/predictions/mdv5a -T path/to/image.jpg
```

### Sagemaker Deployment

Build docker container

```sh
export AWS_ACCOUNT=
export AWS_REGION=

export PII_REGISTRY_NAME=fpe-pii
export PII_IMAGE=${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PII_REGISTRY_NAME}:latest

# aws ecr create-repository --repository-name ${PII_REGISTRY_NAME}
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
docker build -t ${PII_REGISTRY_NAME} .
docker tag ${PII_REGISTRY_NAME}:latest ${PII_IMAGE}
docker push ${PII_IMAGE}
```

----

## Running MegaDetector Locally

Create conda environment from `Microsoft/cameratraps`:

```
conda env create --file lib/cameratraps/environment-detector.yml
conda activate cameratraps-detector
```

Finally, add the submodule repositories to your Python path (whenever you start a new shell).
```
export PYTHONPATH="$PYTHONPATH:$(pwd)/lib/cameratraps:$(pwd)/lib/ai4eutils:$(pwd)/lib/yolov5"
```

**3. Run MegaDetector on a batch of images in a folder.**
```
python lib/cameratraps/detection/run_detector_batch.py model/md_v5a.0.0.pt /path/to/images /path/to/output-640.json --output_relative_filenames --recursive --ncores 4
```

----

## MegaDetector (Amrita's Notes)

### Detecting Personally-Identifiable Information in Photos with MegaDetector

Photos taken at streamflow monitoring sites may contain people, vehicles, or other objects that should be screened before the photos are uploaded to Flow Photo Explorer. The MegaDetector model is an object detector trained to detect objects of 3 classes (person, vehicle, and animal) in camera trap imagery. We can use the latest versions of MegaDetector (v5) to flag images that potentially should not be uploaded.

### Setup

To detect PII in photos using MegaDetector, follow the steps outlined in the [MegaDetector repo](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md).

**1. Downloading the model:**

Model checkpoints for MegaDetector v5a and MegaDetector v5b have been downloaded to the `fpe-model/results/checkpoints` directory. TBD: Should we use DVC to store these files somewhere that isn't this GitHub repo?

**2. Clone relevant repos and add them to your path within the required Python environment:**

The `ai4eutils`, `cameratraps`, and `yolov5` repos have been added as submodules in `fpe-model` and can be found under `fpe-model/third_party`.

To set up the Python environment needed for MegaDetector, use the `fpe-model/third_party/cameratraps/environment-detector.yml` file with `conda`:

```
conda env create --file environment-detector.yml
conda activate cameratraps-detector
```

Finally, add the submodule repositories to your Python path (whenever you start a new shell).
```
export PYTHONPATH="$PYTHONPATH:$(pwd)/third_party/cameratraps:$(pwd)/third_party/ai4eutils:$(pwd)/third_party/yolov5"
```

**3. Run MegaDetector on a batch of images in a folder.**
```
python detection/run_detector_batch.py "{FULL_PATH_TO_FPE-MODEL_REPO}/results/checkpoints/md_v5a.0.0.pt" "/some/image/folder" "{FULL_PATH_TO_FPE-MODEL_REPO}/results/test_output.json" --output_relative_filenames --recursive --checkpoint_frequency 10000
```

The detection results are in the JSON file specified in the third argument to the script.

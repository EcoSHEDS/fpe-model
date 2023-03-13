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
pip install -r lib/yolov5/requirements.txt torch-model-archiver
```

### Model Artifacts

Original and processed artifacts can be fetched directly with DVC:

```
dvc pull
```

Or they can be re-downloaded and processed:

*Step 1*: Fetch megadetector checkpoints (.pt) files from github

```sh
./fetch-mdv5.sh
```

*Step 2*: Convert megadetector files to torchscript using `yolov5`

```sh
python lib/yolov5/export.py --weights models/mdv5/md_v5a.0.0.pt --img 640 --batch 1 --include torchscript
python lib/yolov5/export.py --weights models/mdv5/md_v5b.0.0.pt --img 640 --batch 1 --include torchscript
```

*Step 3*: Convert torchscript to torch model archive (mar)

```sh
torch-model-archiver --model-name mdv5a --version 1.0.0 --serialized-file models/mdv5/md_v5a.0.0.torchscript --extra-files index_to_name.json --handler mdv5_handler.py && mv mdv5a.mar models/mdv5/

torch-model-archiver --model-name mdv5b --version 1.0.0 --serialized-file models/mdv5/md_v5b.0.0.torchscript --extra-files index_to_name.json --handler mdv5_handler.py && mv mdv5b.mar models/mdv5/
```

*Step 4*: Compress for upload to S3 (see `deploy-mdv5.ipynb`)

```sh
cd models/mdv5
tar czvf mdv5a.tar.gz mdv5a.mar
tar czvf mdv5b.tar.gz mdv5b.mar
cd ../..
```

### Run TorchServe Locally

Run torchserve via docker

```sh
./docker-mdv5.sh
# example request
# curl http://127.0.0.1:8080/predictions/mdv5a -T path/to/image.jpg
```

### Sagemaker Deployment

See `deploy-mdv5.ipynb` for deployment instructions

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
python lib/cameratraps/detection/run_detector_batch.py models/mdv5/md_v5a.0.0.pt /path/to/images /path/to/output-640.json --output_relative_filenames --recursive --ncores 4
```


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

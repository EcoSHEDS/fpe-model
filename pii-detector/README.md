FPE Personal Identifying Information (PII) Detector
---------------------------------------------------

Purpose: to create sagemaker endpoint for detecting PII in a single image. This endpoint will be invoked by the FPE image processor whenever a new imageset is uploaded.

References:
- [microsoft/CameraTraps](https://github.com/microsoft/CameraTraps): megadetector model
- [tnc-ca-geo/animl-ml](https://github.com/tnc-ca-geo/animl-ml): machine learning tools for processing camera trap data, used as reference for deploying megadetector model to sagemaker
- [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

[API Docs for Tenserflow Serving Model](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/sagemaker.tensorflow.html#tensorflow-serving-model)

`sagemaker.tensorflow.model.TensorFlowModel(model_data, role, entry_point, ...)`

`model_data` contains refers to a zip file on S3 containing:

```
model1
    |--[model_version_number]
        |--variables
        |--saved_model.pb
model2
    |--[model_version_number]
        |--assets
        |--variables
        |--saved_model.pb
code
    |--inference.py
    |--requirements.txt
```

note: use [Batch Transform](https://github.com/aws/sagemaker-tensorflow-serving-container#creating-a-batch-transform-job) job for large-scale inference

use [yolov5/export.py](https://github.com/tnc-ca-geo/animl-ml/blob/master/api/megadetectorv5/yolov5/export.py) to convert megadetector `md_v5a.0.0.pt` to [SavedModel](https://www.tensorflow.org/guide/saved_model) format

[Sagemaker PyTorch Model Server](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#id4)

use `sagemaker.pytorch.model.PyTorchModel` to load model trained outside of Sagemaker and deploy to endpoint.

```py
from sagemaker import get_execution_role
role = get_execution_role()

pytorch_model = PyTorchModel(model_data='s3://my-bucket/my-path/model.tar.gz'
                             role=role,
                             entry_point='inference.py')

predictor = pytorch_model.deploy(
  instance_type='ml.c4.xlarge',
  initial_instance_count=1
)
```

`model_data` must contain structure:
```
| my_model
|   |--model.pth
|   code
|     |--inference.py
|     |--requirements.txt
```

Not obvious what the inference script should do.

What about using docker to create a custom container for sagemaker.

[AWS | SageMaker | Developer Guide | Deploy Model](https://docs.amazonaws.cn/en_us/sagemaker/latest/dg/deploy-model.html)

Inference options:
- real time (sustained traffic)
- serverless (intermittent)
- batch (offline)
- asynchronous (queue)

Following the `animl-ml` deployment notebook: https://github.com/tnc-ca-geo/animl-ml/blob/master/api/megadetectorv5/sagemaker-serverless-endpoint-with-torchserve/notebook/mdv5_deploy.ipynb

Create a model archive (`mar`) file (see [README](https://github.com/tnc-ca-geo/animl-ml/blob/master/api/megadetectorv5/README.md))

```sh
# convert pt to torchscript using yolov5
conda create -n yolov5 python=3.9
conda activate yolov5
pip install -r ../third_party/yolov5/requirements.txt
python ../third_party/yolov5/export.py --weights ../third_party/assets/md_v5a.0.0.pt --img 640 640 --batch 1
# creates: third_party/assets/md_v5a.0.0.torchscript and third_party/assets/md_v5a.0.0.onnx

# convert torchscript to mar using torch-model-archiver
pip install torchserve torch-model-archiver torch-workflow-archiver
torch-model-archiver --model-name mdv5a --version 1.0.0 --serialized-file ../third_party/assets/md_v5a.0.0.torchscript --extra-files index_to_name.json --handler mdv5a-handler.py
```

Run torch serve

```sh
torchserve --start --model-store ./model_store --no-config-snapshots --models mdv5a=./mdv5a.mar
# requires java
```

Run in docker container

```sh
docker run --rm -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 -v "$(pwd)":/app -it pytorch/torchserve:latest torchserve --start --model-store /app/model_store --no-config-snapshots --models mdv5a=/app/mdv5a.mar
```

```sh
docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -v $(pwd)/model_store:/home/model-server/model-store pytorch/torchserve:latest-cpu
curl -X POST "localhost:8081/models?model_name=mdv5a&url=mdv5a.mar&initial_workers=4"
```

```sh
curl http://127.0.0.1:8080/predictions/mdv5a -T ../img/sample-img-fox.jpg
```

----

## Instructions

Reference: see `lib/animal_detector/mdv5app`

Set working directory to `pii-detector/` sub-directory.

```sh
cd pii-detector
```

Set up conda environment:

```sh
conda create -n fpe-pii-yolov5 python=3.8
conda activate fpe-pii-yolov5
pip install -r lib/yolov5/requirements.txt torch-model-archiver
```

Fetch megadetector checkpoints (.pt) files from github

```sh
./fetch-mdv5.sh
```

Convert megadetector files to torchscript using `yolov5`

```sh
python lib/yolov5/export.py --weights models/mdv5/md_v5a.0.0.pt --img 640 640 --batch 1 --include torchscript
```

Convert torchscript to torch model archive (mar)

```sh
torch-model-archiver --model-name mdv5 --version 1.0.0 --serialized-file models/mdv5/md_v5a.0.0.torchscript --extra-files index_to_name.json --handler mdv5_handler.py
mkdir -p model_store
mv mdv5.mar model_store/mdv5.mar
```

Run torchserve via docker (optional)

```sh
./docker-mdv5.sh
# example request
# curl http://127.0.0.1:8080/predictions/mdv5 -T path/to/image.jpg
```

## Megadetector
# Detecting Personally-Identifiable Information in Photos with MegaDetector
Photos taken at streamflow monitoring sites may contain people, vehicles, or other objects that should be screened before the photos are uploaded to Flow Photo Explorer. The MegaDetector model is an object detector trained to detect objects of 3 classes (person, vehicle, and animal) in camera trap imagery. We can use the latest versions of MegaDetector (v5) to flag images that potentially should not be uploaded.

## Setup
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
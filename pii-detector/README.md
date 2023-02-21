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
pip install 
torchserve --start --model-store ./model_store --no-config-snapshots --models mdv5a=./mdv5a.mar
# requires java
```

Run in docker container

```
docker run --rm -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 -v "$(pwd)":/app -it pytorch/torchserve:latest torchserve --start --model-store /app/model_store --no-config-snapshots --models mdv5a=/app/mdv5a.mar
```

```sh
docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -v $(pwd)/model_store:/home/model-server/model-store pytorch/torchserve:latest-cpu
curl -X POST "localhost:8081/models?model_name=mdv5a&url=mdv5a.mar&initial_workers=4"
```

```
curl http://127.0.0.1:8080/predictions/mdv5a -T ../img/sample-img-fox.jpg
```
from time import strftime, gmtime

import boto3
import sagemaker.session
from sagemaker import ScriptProcessor
from sagemaker.model import Model
from sagemaker.config import NETWORK_CONFIG
from sagemaker.network import NetworkConfig
from sagemaker.processing import ProcessingOutput, ProcessingInput
from sagemaker.workflow import ParameterString
from sagemaker.workflow.functions import Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.workflow.pipeline import Pipeline

from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

from sagemaker.pytorch.model import PyTorchModel
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.transformer import Transformer
from sagemaker.inputs import TransformInput
from sagemaker.workflow.steps import TransformStep

# Variables

AWS_ACCOUNT_ID = boto3.client("sts").get_caller_identity().get("Account")
AWS_REGION = boto3.session.Session().region_name
ECR_R_REPOSITORY_NAME = "sagemaker_pipeline_r"
ECR_R_REPOSITORY_TAG = ":latest"
ECR_REPOSITORY = "{}.dkr.ecr.{}.amazonaws.com/{}"
ECR_R_REPOSITORY = ECR_REPOSITORY.format(
    AWS_ACCOUNT_ID,
    AWS_REGION,
    ECR_R_REPOSITORY_NAME + ECR_R_REPOSITORY_TAG
)


NETWORK_CONFIG = NetworkConfig(
    enable_network_isolation=False,
    security_group_ids=["sg-044c8a541b32a89b6"],
    subnets=["subnet-0fe130a3b16016bf2"],
)
pipeline_session = PipelineSession()
pipeline_name = f"USGSPipeline"

# Pipeline Parameters
param_model_type = ParameterString(
    name="ModelType",
    default_value="NONE"
)

param_variable_code = ParameterString(
    name="VariableCode",
    default_value="FLOW_CFS"
)

param_station_id = ParameterString(
    name="StationID",
    default_value="29"
)

param_min_hour = ParameterString(
    name="MinHour",
    default_value="7"
)

param_max_hour = ParameterString(
    name="MaxHour",
    default_value="18"
)

param_min_month = ParameterString(
    name="MinMonth",
    default_value="1"
)

param_max_month = ParameterString(
    name="MaxMonth",
    default_value="12"
)

param_n_epoch = ParameterString(
    name="TrainingEpoch",
    default_value="1"
)

param_learning_rate = ParameterString(
    name="LearningRate",
    default_value="0.001"
)

# STEP 1

station_images_annotations_processor = ScriptProcessor(
    command=["Rscript"],
    image_uri=ECR_R_REPOSITORY,
    role="arn:aws:iam::010438494535:role/Dev-USGS-FPE-Environment",
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=pipeline_session,
    network_config=NETWORK_CONFIG,
)

step_station_images_annotations = ProcessingStep(
    name="CreateStationImages",
    step_args=station_images_annotations_processor.run(
        job_name="CreateStationImages-{}".format(strftime("%d-%H-%M-%S", gmtime())),
        inputs=[],
        outputs=[
            ProcessingOutput(output_name="annotations", source="/opt/ml/processing/datasets/RANK-FLOW/"),
        ],
        code="r/rank-dataset.R",
        arguments=[
            "-s", param_station_id,
            "-v", param_variable_code,
            "-d", "/opt/ml/processing",
            "-o",
            "RANK-FLOW"
        ]
    )
)

# STEP 2

create_model_training_dataset_processor = ScriptProcessor(
    command=["Rscript"],
    image_uri=ECR_R_REPOSITORY,
    role="arn:aws:iam::010438494535:role/Dev-USGS-FPE-Environment",
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=pipeline_session,
    network_config=NETWORK_CONFIG,
)

create_model_training_dataset_step = ProcessingStep(
    name="CreateModelTrainingDataset",
    step_args=station_images_annotations_processor.run(
        job_name="CreateModelTrainingDataset-{}".format(strftime("%d-%H-%M-%S", gmtime())),
        inputs=[
            ProcessingInput(source=step_station_images_annotations.properties.ProcessingOutputConfig.Outputs[
                "annotations"
            ].S3Output.S3Uri, destination="/opt/ml/processing/datasets/RANK-FLOW")
        ],
        outputs=[
            ProcessingOutput(output_name="dataset", source="/opt/ml/processing/models/RANK-FLOW/input")
        ],
        code="r/rank-input.R",
        arguments=[
            "-s", param_station_id,
            "-v", param_variable_code,
            "-d", "/opt/ml/processing",
            "-D", "RANK-FLOW",
            "--overwrite",
            "RANK-FLOW"
        ]
    )
)

output_path = "s3://usgstest-jmfpe-dev-model/rank/models/RANK-FLOW-{}/".format(strftime("%d-%H-%M-%S", gmtime()))
# STEP 3
pytorch_estimator = PyTorch(
        entry_point="train.py",
        source_dir="src",
        py_version="py39",
        framework_version="1.13.1",
        role="arn:aws:iam::010438494535:role/Dev-USGS-FPE-Environment",
        instance_count=1,
        instance_type="ml.p3.2xlarge",
        volume_size=100,
        hyperparameters={
            "epochs": param_n_epoch
        },
        output_path=output_path+"jobs",
        checkpoint_s3_uri=output_path+"checkpoints",
        code_location=output_path+"jobs",
        disable_output_compression=False,
        sagemaker_session=pipeline_session,
)

training_arguments = pytorch_estimator.fit(
    inputs = {
        "images": TrainingInput(
        s3_data = Join(on='/', values= [create_model_training_dataset_step.properties.ProcessingOutputConfig.Outputs['dataset'].S3Output.S3Uri, 'manifest.json']),
        s3_data_type = "ManifestFile",
        input_mode = "File"
        ),
        "values": create_model_training_dataset_step.properties.ProcessingOutputConfig.Outputs['dataset'].S3Output.S3Uri
    }
)

training_step = TrainingStep(
    name ="TrainPytorchModel",
    step_args = training_arguments,

)

# STEP 4
model = PyTorchModel(
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session = pipeline_session,
    role="arn:aws:iam::010438494535:role/Dev-USGS-FPE-Environment",
    source_dir="src",
    entry_point="transform.py",
    py_version="py39",
    framework_version="1.13.1",
)

# step_create_model = ModelStep(
#     name="CreatingModel",
#     step_args=model.create(instance_type="ml.m5.large")
# )

# transformer = Transformer(
#     model_name=step_create_model.properties.ModelName,
#     instance_type="ml.c5.xlarge",
#     instance_count=1,
#     output_path=Join(on='/', values= [create_model_training_dataset_step.properties.ProcessingOutputConfig.Outputs['dataset'].S3Output.S3Uri, 'transform']),
#     sagemaker_session=pipeline_session,
# )
transformer = model.transformer(
    instance_count=1,
    instance_type="ml.c5.xlarge",
    output_path=Join(on='/', values=[
        create_model_training_dataset_step.properties.ProcessingOutputConfig.Outputs['dataset'].S3Output.S3Uri,
        'transform'
    ]),
    sagemaker_session=pipeline_session)

step_transform = TransformStep(
    name="RuningInference",
    transformer=transformer,
    inputs=TransformInput(
        data=Join(on='/', values= [create_model_training_dataset_step.properties.ProcessingOutputConfig.Outputs['dataset'].S3Output.S3Uri, 'manifest.json']),
        data_type="ManifestFile", content_type="image/jpg"
    )
)

# Step 5: Run Transform Merge

'''
step_lambda = LambdaStep(
    name="ProcessBatchInferenceOutput",
    lambda_func=Lambda(
        function_arn="arn:aws:lambda:us-east-1:010438494535:function:jmfpe-dev-lambda-models"
    )

)
'''

# Create Pipeline
pipeline = Pipeline(
    name = pipeline_name,
    parameters = [
        param_model_type,
        param_variable_code,
        param_station_id,
        param_min_hour,
        param_max_hour,
        param_min_month,
        param_max_month,
        param_n_epoch,
        param_learning_rate
    ],
    steps = [step_station_images_annotations, create_model_training_dataset_step, training_step, step_transform ],
    sagemaker_session=pipeline_session,
)

print(pipeline.definition())
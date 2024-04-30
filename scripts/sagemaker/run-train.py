import argparse
import boto3
import time
import json
import os
import pandas as pd
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from io import StringIO

from utils import get_batch_creds, timestamp

# aws parameters
AWS_PROFILE="conte-prod"
AWS_REGION="us-west-2"
JOB_ROLE_ARN="arn:aws:iam::694155575325:role/fpe-prod-sagemaker-execution-role"

# # job parameters
# root_dir = f"/mnt/d/fpe/rank"
# station_id = 29
# #dataset = "FLOW-20240327"
# model = "RANK-FLOW-20240402"


def run_train (station_id, model_code, directory):
    print(f"run_train: {station_id} {model_code} {directory}")
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    s3 = session.client("s3")

    creds = get_batch_creds(session, JOB_ROLE_ARN)
    sm_boto_session = boto3.Session(
        aws_access_key_id=creds['AccessKeyId'],
        aws_secret_access_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken'],
        region_name=AWS_REGION
    )
    sm_session = sagemaker.Session(boto_session = sm_boto_session)

    job_name = f"fpe-rank-{timestamp()}"

    model_dir = f"{directory}/{station_id}/models/{model_code}"
    if not os.path.exists(model_dir):
        raise Exception(f"model_dir not found ({model_dir})")

    with open(os.path.join(model_dir, "job.txt"), "w") as f:
        f.write(job_name)

    model_bucket = "usgs-chs-conte-prod-fpe-models"
    storage_bucket = "usgs-chs-conte-prod-fpe-storage"

    model_key = f"rank/{station_id}/models/{model_code}"
    input_key = f"{model_key}/input"
    jobs_key = f"{model_key}/jobs"
    checkpoints_key = f"{model_key}/checkpoints"
    transform_key = f"{model_key}/transform"

    s3_input_path = f"s3://{model_bucket}/{input_key}"
    s3_output_path = f"s3://{model_bucket}/{jobs_key}"
    s3_checkpoint_path = f"s3://{model_bucket}/{checkpoints_key}"
    s3_transform_path = f"s3://{model_bucket}/{transform_key}"

    # upload input to s3
    for subdir, dirs, files in os.walk(f"{model_dir}/input"):
        for file in files:
            s3_key = f"{input_key}/{file}"
            print(f"uploading: {file} -> {s3_key}")
            s3.upload_file(Filename=os.path.join(subdir, file), Bucket=model_bucket, Key=s3_key)

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="src",
        py_version="py39",
        framework_version="1.13.1",
        role="arn:aws:iam::694155575325:role/fpe-prod-sagemaker-execution-role",
        instance_count=1,
        instance_type="ml.p3.2xlarge",
        volume_size=100,
        hyperparameters={
            "epochs": 20
        },
        output_path=s3_output_path,
        checkpoint_s3_uri=s3_checkpoint_path,
        code_location=s3_output_path,
        disable_output_compression=False,
        sagemaker_session=sm_session
    )

    training_input = TrainingInput(
        s3_data = f"{s3_input_path}/manifest.json",
        s3_data_type = "ManifestFile",
        input_mode = "File"
    )

    estimator.fit({ "images": training_input, "values": s3_input_path }, job_name=job_name, wait=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--station-id", type=str)
    parser.add_argument("--model-code", type=str)
    parser.add_argument("--directory", type=str)

    args = parser.parse_args()
    print(args)

    run_train(args.station_id, args.model_code, args.directory)


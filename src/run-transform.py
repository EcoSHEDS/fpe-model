import argparse
import boto3
import time
import json
import os
from datetime import datetime
import pandas as pd
import sagemaker
from sagemaker.pytorch.model import PyTorchModel
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


def run_transform (station_id, model_code, directory):
    print(f"run_transform: {station_id} {model_code} {directory}")

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

    model_dir = f"{directory}/{station_id}/models/{model_code}"
    if not os.path.exists(model_dir):
        raise Exception(f"model_dir not found ({model_dir})")

    print(f"model_dir: {model_dir}")

    with open(os.path.join(model_dir, "job.txt"), "r") as f:
        job_name = f.readline()

    print(f"job_name: {job_name}")

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

    transform_images_file = f"{model_dir}/input/images.csv"
    transform_images = pd.read_csv(transform_images_file)
    print(f"# images: {len(transform_images)}")

    transform_images_key = f"{transform_key}/images.csv"
    print(f"uploading: {transform_images_file} -> {transform_images_key}")
    s3.upload_file(Filename=transform_images_file, Bucket=model_bucket, Key=transform_images_key)

    manifest = transform_images['filename'].to_list()
    manifest.insert(0, {"prefix": f"s3://{storage_bucket}/"})

    manifest_key = f"{transform_key}/manifest.json"
    body = json.dumps(manifest)
    print(f"uploading transform manifest: {manifest_key} (n = {len(manifest) - 1})")
    s3.put_object(Bucket=model_bucket, Key=manifest_key, Body=body)

    model_path = f"{s3_output_path}/{job_name}/output/model.tar.gz"
    print(f"model_path: {model_path}")

    print(f"creating model: {model_path}")
    pytorch_model = PyTorchModel(
        model_data=model_path,
        role="arn:aws:iam::694155575325:role/fpe-prod-sagemaker-execution-role",
        py_version="py38",
        framework_version="1.12",
        source_dir="src/",
        entry_point="transform.py",
        sagemaker_session = sm_session
    )

    print(f"creating transformer: {s3_transform_path}")
    transformer = pytorch_model.transformer(
        instance_count=1,
        instance_type="ml.c5.xlarge",
        output_path=s3_transform_path,
        max_payload=20
    )

    print(f"starting transform job: {job_name}")
    transformer.transform(
        data=f"{s3_transform_path}/manifest.json",
        data_type="ManifestFile",
        content_type="image/jpg",
        job_name=f"fpe-rank-transform-{station_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        wait=False,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--station-id", type=str)
    parser.add_argument("--model-code", type=str)
    parser.add_argument("--directory", type=str)

    args = parser.parse_args()
    print(args)

    run_transform(args.station_id, args.model_code, args.directory)


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

def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def get_batch_creds(session, role_arn):
    sts = session.client("sts")
    response = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName=f"fpe-sagemaker-session--{timestamp()}"
    )
    return response['Credentials']

def create_session(aws_profile=None, aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None, region_name=None):
    """Create a boto3 session using either profile or access keys."""
    if aws_profile:
        return boto3.Session(profile_name=aws_profile, region_name=region_name)
    else:
        return boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name
        )

def create_training_job(
    model_code,
    model_dir,
    aws_region,
    job_role_arn,
    model_bucket,
    storage_bucket,
    instance_type,
    instance_count,
    volume_size,
    epochs,
    framework_version,
    py_version,
    aws_profile=None,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None
):
    print(f"create_training_job: {model_code}")
    
    # Create session using either profile or access keys
    session = create_session(
        aws_profile=aws_profile,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=aws_region
    )
    s3 = session.client("s3")

    creds = get_batch_creds(session, job_role_arn)
    sm_boto_session = boto3.Session(
        aws_access_key_id=creds['AccessKeyId'],
        aws_secret_access_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken'],
        region_name=aws_region
    )
    sm_session = sagemaker.Session(boto_session=sm_boto_session)

    job_name = f"fpe-model-{timestamp()}"

    if not os.path.exists(model_dir):
        raise Exception(f"model_dir not found ({model_dir})")

    with open(os.path.join(model_dir, "job.txt"), "w") as f:
        f.write(job_name)

    model_key = f"experiments/{model_code}"
    input_key = f"{model_key}/input"
    jobs_key = f"{model_key}/jobs"
    checkpoints_key = f"{model_key}/checkpoints"
    transform_key = f"{model_key}/transform"

    s3_input_path = f"s3://{model_bucket}/{input_key}"
    s3_output_path = f"s3://{model_bucket}/{jobs_key}"
    s3_checkpoint_path = f"s3://{model_bucket}/{checkpoints_key}"
    s3_transform_path = f"s3://{model_bucket}/{transform_key}"

    # upload input to s3
    try:
        for subdir, dirs, files in os.walk(f"{model_dir}/input"):
            for file in files:
                s3_key = f"{input_key}/{file}"
                print(f"uploading: {file} -> {s3_key}")
                s3.upload_file(Filename=os.path.join(subdir, file), Bucket=model_bucket, Key=s3_key)
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="src",
        py_version=py_version,
        framework_version=framework_version,
        role=job_role_arn,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size=volume_size,
        hyperparameters={
            "epochs": epochs
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

    # Required arguments
    parser.add_argument("--model-code", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--aws-region", type=str, required=True, help="AWS region")
    parser.add_argument("--job-role-arn", type=str, required=True, help="AWS IAM role ARN")

    # AWS authentication (either profile or access keys)
    auth_group = parser.add_mutually_exclusive_group(required=True)
    auth_group.add_argument("--aws-profile", type=str, help="AWS profile name")
    auth_group.add_argument("--aws-access-key-id", type=str, help="AWS access key ID")
    
    # Optional AWS credentials (required if using access key)
    parser.add_argument("--aws-secret-access-key", type=str, help="AWS secret access key")
    parser.add_argument("--aws-session-token", type=str, help="AWS session token")

    # Optional arguments with defaults
    parser.add_argument("--model-bucket", type=str, default="usgs-chs-conte-prod-fpe-models", help="S3 bucket for model artifacts")
    parser.add_argument("--storage-bucket", type=str, default="usgs-chs-conte-prod-fpe-storage", help="S3 bucket for storage")
    parser.add_argument("--instance-type", type=str, default="ml.p3.2xlarge", help="SageMaker instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances to use")
    parser.add_argument("--volume-size", type=int, default=100, help="Size of EBS volume in GB")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--framework-version", type=str, default="2.4.0", help="PyTorch framework version")
    parser.add_argument("--py-version", type=str, default="py311", help="Python version")

    args = parser.parse_args()

    # Validate AWS credentials if using access key
    if args.aws_access_key_id and not args.aws_secret_access_key:
        parser.error("--aws-secret-access-key is required when using --aws-access-key-id")

    print(args)

    # Add validation for required input files
    input_dir = os.path.join(args.model_dir, "input")
    if not os.path.exists(input_dir):
        raise Exception(f"Input directory not found: {input_dir}")
    if not os.path.exists(os.path.join(input_dir, "manifest.json")):
        raise Exception(f"manifest.json not found in {input_dir}")

    create_training_job(
        args.model_code,
        args.model_dir,
        args.aws_region,
        args.job_role_arn,
        args.model_bucket,
        args.storage_bucket,
        args.instance_type,
        args.instance_count,
        args.volume_size,
        args.epochs,
        args.framework_version,
        args.py_version,
        aws_profile=args.aws_profile,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        aws_session_token=args.aws_session_token
    )


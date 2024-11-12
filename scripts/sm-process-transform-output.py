import argparse
import boto3
import subprocess
import json
import os
import pandas as pd

# aws parameters
AWS_PROFILE="conte-prod"
AWS_REGION="us-west-2"
JOB_ROLE_ARN="arn:aws:iam::694155575325:role/fpe-prod-sagemaker-execution-role"

# # job parameters
# root_dir = f"/mnt/d/fpe/rank"
# station_id = 29
# #dataset = "FLOW-20240327"
# model = "RANK-FLOW-20240402"

def run_transform_merge (station_id, model_code, directory, job_size=5000):
    print(f"run_transform_merge: {station_id} {model_code} {directory}")

    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    s3 = session.client("s3")
    lambda_client = session.client("lambda")

    model_dir = f"{directory}/{station_id}/models/{model_code}"
    if not os.path.exists(model_dir):
        raise Exception(f"model_dir not found ({model_dir})")

    print(f"model_dir: {model_dir}")

    model_bucket = "usgs-chs-conte-prod-fpe-models"
    model_key = f"rank/{station_id}/models/{model_code}"
    jobs_key = f"{model_key}/jobs"
    transform_key = f"{model_key}/transform"
    transform_images_file = f"{model_dir}/input/images.csv"
    transform_images_key = f"{transform_key}/images.csv"

    df = pd.read_csv(transform_images_file)
    skip = 0
    while skip < len(df):
        payload = {
            "action": "process_transform_output",
            "bucket_name": model_bucket,
            "data_file": transform_images_key,
            "data_prefix": transform_key,
            "output_prefix": transform_key,
            "n": job_size,
            "skip": skip
        }
        print(f"invoke: skip={skip}, n={job_size} ({skip} to {skip + job_size - 1})")
        lambda_client.invoke(
            FunctionName="fpe-prod-lambda-models",
            InvocationType="Event",
            Payload=json.dumps(payload)
        )
        skip += job_size
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--station-id", type=str)
    parser.add_argument("--model-code", type=str)
    parser.add_argument("--directory", type=str)

    args = parser.parse_args()
    print(args)

    run_transform_merge(args.station_id, args.model_code, args.directory)


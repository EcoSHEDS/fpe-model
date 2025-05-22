import argparse
import boto3
import subprocess
import os
import pandas as pd
from io import StringIO

# aws parameters
AWS_PROFILE="conte-prod"
AWS_REGION="us-west-2"
JOB_ROLE_ARN="arn:aws:iam::694155575325:role/fpe-prod-sagemaker-execution-role"

# # job parameters
# root_dir = f"/mnt/d/fpe/rank"
# station_id = 29
# #dataset = "FLOW-20240327"
# model = "RANK-FLOW-20240402"

def run_transform_predictions (station_id, model_code, directory, job_size = 5000):
    print(f"run_transform_merge: {station_id} {model_code} {directory}")

    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    s3 = session.client("s3")

    model_dir = f"{directory}/{station_id}/models/{model_code}"
    if not os.path.exists(model_dir):
        raise Exception(f"model_dir not found ({model_dir})")

    print(f"model_dir: {model_dir}")

    transform_path = f"{model_dir}/transform"
    if not os.path.exists(transform_path):
        os.makedirs(transform_path)

    model_bucket = "usgs-chs-conte-prod-fpe-models"
    model_key = f"rank/{station_id}/models/{model_code}"
    jobs_key = f"{model_key}/jobs"
    transform_key = f"{model_key}/transform"
    transform_images_file = f"{model_dir}/input/images.csv"

    with open(os.path.join(model_dir, "job.txt"), "r") as f:
        job_name = f.readline()

    print(f"job: {job_name}")

    output_dir = f"{model_dir}/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f"{output_dir}/output.tar.gz"

    output_key = f"{jobs_key}/{job_name}/output/output.tar.gz"
    print(f"downloading: s3://{model_bucket}/{output_key} -> {output_file}")
    s3.download_file(Bucket=model_bucket, Key=output_key, Filename=output_file)

    model_file = f"{output_dir}/model.tar.gz"
    model_key = f"{jobs_key}/{job_name}/output/model.tar.gz"
    print(f"downloading: s3://{model_bucket}/{model_key} -> {model_file}")
    s3.download_file(Bucket=model_bucket, Key=model_key, Filename=model_file)

    print(f"extracting: {output_file}")
    subprocess.run(["tar", "-xzvf", output_file, "-C", output_dir])

    df = pd.read_csv(transform_images_file)
    keys = [f"{transform_key}/predictions-{skip:05d}-{(skip + job_size - 1):05d}.csv" for skip in range(0, len(df), job_size)]

    dfs = []
    for key in keys:
        print(f"downloading: {key}")
        csv_obj = s3.get_object(Bucket=model_bucket, Key=key)
        csv_data = csv_obj['Body'].read().decode('utf-8')
        dfs.append(pd.read_csv(StringIO(csv_data)))

    df = pd.concat(dfs, ignore_index=True)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    output_key = f"{transform_key}/predictions.csv"
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=model_bucket, Key=output_key)

    print(f"saving: {transform_path}/predictions.csv")
    df.to_csv(f"{transform_path}/predictions.csv", index=False)
    df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--station-id", type=str)
    parser.add_argument("--model-code", type=str)
    parser.add_argument("--directory", type=str)

    args = parser.parse_args()
    print(args)

    run_transform_predictions(args.station_id, args.model_code, args.directory)


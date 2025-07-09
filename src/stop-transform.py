import argparse
import boto3
import sagemaker

from utils import get_batch_creds

# aws parameters
AWS_PROFILE="conte-prod"
AWS_REGION="us-west-2"
JOB_ROLE_ARN="arn:aws:iam::694155575325:role/fpe-prod-sagemaker-execution-role"

def stop_transform (job_name):
    print(f"stop_transform: {job_name}")
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    creds = get_batch_creds(session, JOB_ROLE_ARN)
    sm_boto_session = boto3.Session(
        aws_access_key_id=creds['AccessKeyId'],
        aws_secret_access_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken'],
        region_name=AWS_REGION
    )
    sm_session = sagemaker.Session(boto_session = sm_boto_session)
    sm_session.stop_transform_job(job_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--job-name", type=str)

    args = parser.parse_args()
    print(args)

    stop_transform(args.job_name)


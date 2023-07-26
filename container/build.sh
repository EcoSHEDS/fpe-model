#!/usr/bin/env bash

image=$1

dockerfile=${2:-Dockerfile}

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)
region=${region:-us-east-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
echo "ECR image fullname: ${fullname}"

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    echo "Creating ECF repository: ${image}"
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
export _DOCKER_REPO="$(aws ecr get-authorization-token --output text  --query 'authorizationData[].proxyEndpoint')"
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin $_DOCKER_REPO

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "763104351884.dkr.ecr.${region}.amazonaws.com"

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build -f ${dockerfile} -t ${image} . --build-arg REGION=${region}
docker tag ${image} ${fullname}
docker push ${fullname}
# an alternative and simplified command that can be used instead of the code above is
# sm-docker build ${1}
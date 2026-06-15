# Containerized AWS Batch inference (single imageset)

A self-contained Docker container that scores **one imageset** with a trained RankNet model
and writes a drop-in `predictions.csv` to S3. It runs as a single AWS Batch job and replaces
the SageMaker **transform → merge → predictions** chain (and its Lambda merge step) for
incremental imageset scoring, where the workload is many jobs each up to tens of thousands of
images.

- **Entrypoint:** [`src/predict-imageset.py`](../src/predict-imageset.py)
- **Core logic:** [`src/fpe_imageset.py`](../src/fpe_imageset.py) (DB fetch + filters) and
  [`src/fpe_inference.py`](../src/fpe_inference.py) (batched inference)
- **Image:** [`Dockerfile`](../Dockerfile) + [`requirements-batch.txt`](../requirements-batch.txt)

## What the job does

Given a station id, model code, and imageset uuid, the container:

1. reads the model's `station.json` + `rank-input.json` from the model bucket (timezone + the
   daytime/seasonal filters the model was built with);
2. looks up the model's UUID in the Postgres `models` table (`station_id` + `code`) and pulls
   `models/{uuid}/model.tar.gz` from the storage bucket;
3. queries Postgres for the imageset's `DONE` images, applies the model's hour/month filters in
   the station's local timezone, and validates the imageset belongs to the station;
4. runs **batched** inference on CPU using a `DataLoader` that overlaps S3 image download +
   decode with model compute;
5. writes `predictions.csv` with columns `split,image_id,timestamp,filename,url,value,score` to:
   ```
   s3://usgs-chs-conte-prod-fpe-models/rank/{station_id}/models/{model_code}/imagesets/{imageset_uuid}/transform/predictions.csv
   ```
   — the same `predictions.csv` schema the station-wide SageMaker pipeline produces, namespaced
   under the imageset uuid.

Any failure (model not found, no `DONE` images, empty after filtering, etc.) exits non-zero so
Batch marks the job **FAILED**.

### Score parity

Scores match the SageMaker serving path to within float tolerance. The container reuses
`transform.model_fn` for the model load and mirrors `transform.input_fn`/`predict_fn` exactly:
PIL `Image.open` + `ToTensor` (no `.convert`), then `model.module.transforms['eval']`, then
`model.module.forward_single`. BatchNorm runs in `eval()` mode (running stats), so scoring N
images in a batch is numerically identical to the per-image loop. See the parity test:
[`tests/test_inference_parity.py`](../tests/test_inference_parity.py).

## Parameters

Each parameter may be passed as a CLI arg **or** an environment variable (Batch can use either;
the CLI arg wins when both are set).

| CLI arg | Env var | Required | Default | Description |
|---|---|---|---|---|
| `--station-id` | `STATION_ID` | yes | — | FPE station id |
| `--model-code` | `MODEL_CODE` | yes | — | model code (e.g. `RANK-FLOW-20240709`) |
| `--imageset-uuid` | `IMAGESET_UUID` | yes | — | imageset UUID to score (the `imagesets.uuid`, as in storage paths `imagesets/{uuid}/...`) |
| `--batch-size` | `BATCH_SIZE` | no | `32` | inference batch size |
| `--num-workers` | `NUM_WORKERS` | no | `4` | `DataLoader` workers (S3 download/decode parallelism) |

Database connection. Preferred: set `FPE_DB_SECRET` to a Secrets Manager secret name; the
container fetches and parses it (using the Batch job role's `SecretsManagerReadWrite`). The
secret JSON may use either `database`/`user` or `dbname`/`username` for those two fields. If
`FPE_DB_SECRET` is unset, the container falls back to discrete env vars:

| Env var | Description |
|---|---|
| `FPE_DB_SECRET` | Secrets Manager secret name with DB credentials (preferred) |
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` | discrete fallback when no secret is set |

Optional bucket overrides (default to the prod buckets):

| Env var | Default | Description |
|---|---|---|
| `FPE_S3_STORAGE_BUCKET` | `usgs-chs-conte-prod-fpe-storage` | images + deployed `model.tar.gz` |
| `FPE_S3_MODEL_BUCKET` | `usgs-chs-conte-prod-fpe-models` | model `input/` config + predictions output |

`AWS_REGION` is optional (defaults to `us-west-2`).

## Build and push to ECR

```sh
ACCOUNT=694155575325
REGION=us-west-2
REPO=fpe-predict
PROFILE=conte-prod

# one-time: create the ECR repo
aws ecr create-repository --repository-name "${REPO}" --region "${REGION}" --profile "${PROFILE}"

# authenticate docker to ECR
aws ecr get-login-password --region "${REGION}" --profile "${PROFILE}" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

# build, tag, push (run from the repo root)
docker build --platform linux/amd64 -t "${REPO}" .
docker tag "${REPO}:latest" "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest"
docker push "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest"
```

The base image is `python:3.9-slim` with CPU-only torch/torchvision wheels (no CUDA). AWS Batch
runs x86_64 instances, so always build for `linux/amd64`. On an Apple Silicon (arm64) Mac the
`--platform linux/amd64` flag is required; the image still builds and runs there under emulation
(slow) for smoke tests, but runs natively on Batch.

The build pre-caches the torchvision ResNet-18 weights (`TORCH_HOME=/opt/fpe/.torch`) so
`model_fn`'s `pretrained=True` needs no internet at runtime — important if the Batch compute
environment has no egress.

## Run locally (smoke test)

```sh
docker run --rm \
  -e STATION_ID=29 -e MODEL_CODE=RANK-FLOW-20240709 -e IMAGESET_UUID=e8d465f6-5784-4231-967f-9000428e9748 \
  -e DB_HOST=... -e DB_PORT=5432 -e DB_NAME=... -e DB_USER=... -e DB_PASSWORD=... \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_SESSION_TOKEN=... \
  -e AWS_REGION=us-west-2 \
  fpe-predict
```

You can also run the entrypoint without Docker (needs `requirements.txt` + torch installed):

```sh
STATION_ID=29 MODEL_CODE=RANK-FLOW-20240709 IMAGESET_UUID=e8d465f6-5784-4231-967f-9000428e9748 \
DB_HOST=... DB_PORT=5432 DB_NAME=... DB_USER=... DB_PASSWORD=... \
AWS_PROFILE=conte-prod \
python src/predict-imageset.py
```

## AWS Batch job definition contract

The compute environment, job queue, and IAM are infra-managed; this repo provides only the
container and the code it runs. The job definition must supply:

- **Container image:** the ECR URI pushed above.
- **Command / environment:** `STATION_ID`, `MODEL_CODE`, `IMAGESET_UUID` (and optionally
  `BATCH_SIZE`, `NUM_WORKERS`). These can be `containerProperties.command` overrides or
  `environment` entries; the entrypoint accepts either form.
- **Database credentials:** set `FPE_DB_SECRET` to the DB secret name; the container fetches it
  at runtime via the **job role** (which needs `secretsmanager:GetSecretValue`). No job-definition
  `secrets`/`valueFrom` block is required, so the **execution role** needs no secrets permission.
- **Buckets (optional):** `FPE_S3_STORAGE_BUCKET`, `FPE_S3_MODEL_BUCKET` (default to the prod buckets).
- **Compute:** CPU only — Fargate is the intended launch type. No GPU resource is required.

### IAM (Batch *job* role)

The role the container assumes needs:

| Action | Resource | Why |
|---|---|---|
| `s3:GetObject` | `arn:aws:s3:::usgs-chs-conte-prod-fpe-storage/models/*` | download `model.tar.gz` |
| `s3:GetObject` | `arn:aws:s3:::usgs-chs-conte-prod-fpe-storage/*` | download imageset images |
| `s3:GetObject` | `arn:aws:s3:::usgs-chs-conte-prod-fpe-models/rank/*` | read `station.json` / `rank-input.json` |
| `s3:PutObject` | `arn:aws:s3:::usgs-chs-conte-prod-fpe-models/rank/*` | write `predictions.csv` |
| `secretsmanager:GetSecretValue` | the DB secret ARN | read `FPE_DB_SECRET` at runtime |

Plus network reachability to the Postgres host (security group / subnet). The existing FPE Batch
job role already grants `AmazonS3FullAccess` + `SecretsManagerReadWrite`, which covers all of the
above — see the CloudFormation section below for reusing it.

## Notes / tuning

- **Throughput:** S3 download is usually the bottleneck for this small model, so `--num-workers`
  (download concurrency) matters most; `--batch-size` trades memory for CPU forward-pass efficiency.
  Per-image download/decode failures are logged and recorded as `score=NaN` rather than failing the
  whole job.
- **S3 throttling:** the S3 client uses adaptive retries; keep `num_workers` bounded on very
  large imagesets.
- **Shared memory (Fargate):** `run_inference` sets PyTorch's `file_system` tensor-sharing
  strategy, so the `DataLoader` workers don't depend on `/dev/shm`. This is **required on
  Fargate**, whose `/dev/shm` is a fixed ~64 MB that cannot be enlarged (no
  `linuxParameters.sharedMemorySize` on Fargate; `--shm-size` is local-docker only). Without it,
  workers die with "Unexpected bus error … insufficient shared memory (shm)".
- **Timeout:** size the Batch job timeout to expected throughput for tens-of-thousands-image
  imagesets.

## CloudFormation (reuse the existing FPE Batch stack)

The existing FPE Batch CloudFormation stack (Fargate compute environment, job queue, service /
execution / job roles, and the failed-job EventBridge → SNS rule) is workload-agnostic and can be
reused as-is. The job role already grants `AmazonS3FullAccess` + `SecretsManagerReadWrite`, and the
failed-job rule matches **all** Batch `FAILED` events, so predict jobs get failure alerts for free.
Only **two new resources** are needed (plus one new parameter for the model bucket); model them on
the existing `…Pii` job.

Add to `Parameters`:

```json
"modelBucketName": {
  "Description": "Name of S3 model bucket (model input config + predictions output)",
  "Type": "String"
}
```

Add to `Resources`:

```json
"RepositoryPredict": {
  "Type": "AWS::ECR::Repository",
  "Properties": {
    "RepositoryName": { "Fn::Join": ["-", [{ "Ref": "appName" }, { "Ref": "env" }, "batch-predict"]] }
  }
},
"JobDefinitionPredict": {
  "Type": "AWS::Batch::JobDefinition",
  "Properties": {
    "Type": "container",
    "JobDefinitionName": { "Fn::Join": ["-", [{ "Ref": "appName" }, { "Ref": "env" }, "batch-job-definition-predict"]] },
    "PlatformCapabilities": ["FARGATE"],
    "ContainerProperties": {
      "Image": { "Fn::Sub": "${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/${RepositoryPredict}:latest" },
      "ExecutionRoleArn": { "Fn::GetAtt": ["ExecutionRole", "Arn"] },
      "JobRoleArn": { "Fn::GetAtt": ["JobRole", "Arn"] },
      "FargatePlatformConfiguration": { "PlatformVersion": "LATEST" },
      "ResourceRequirements": [
        { "Type": "MEMORY", "Value": 8192 },
        { "Type": "VCPU", "Value": 4.0 }
      ],
      "Command": ["--help"],
      "NetworkConfiguration": { "AssignPublicIp": "ENABLED" },
      "Environment": [
        { "Name": "AWS_REGION", "Value": { "Ref": "AWS::Region" } },
        { "Name": "FPE_DB_SECRET", "Value": { "Ref": "dbSecretName" } },
        { "Name": "FPE_S3_STORAGE_BUCKET", "Value": { "Ref": "storageBucketName" } },
        { "Name": "FPE_S3_MODEL_BUCKET", "Value": { "Ref": "modelBucketName" } }
      ]
    },
    "RetryStrategy": { "Attempts": 1 }
  }
}
```

Optionally add to `Outputs` (mirrors the processor/PII outputs):

```json
"JobDefinitionNamePredict": {
  "Description": "Predict job definition name",
  "Value": { "Fn::Join": ["-", [{ "Ref": "appName" }, { "Ref": "env" }, "batch-job-definition-predict"]] }
},
"RepositoryUrlPredict": {
  "Description": "Batch predict repository URL",
  "Value": { "Fn::Sub": "${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/${RepositoryPredict}" }
}
```

`Command: ["--help"]` is just a safe default; per-imageset runs pass the real args via
`containerOverrides` at submit time:

```sh
aws batch submit-job \
  --job-name "predict-<imageset-uuid>" \
  --job-queue   "${appName}-${env}-batch-job-queue" \
  --job-definition "${appName}-${env}-batch-job-definition-predict" \
  --container-overrides '{"command":["--station-id","29","--model-code","RANK-FLOW-20240410","--imageset-uuid","<imageset-uuid>"]}' \
  --region us-west-2 --profile conte-prod
```

Sizing: 4 vCPU / 8 GB is a reasonable Fargate default for this small model; raise vCPU/memory and
`NUM_WORKERS` together if S3 download throughput needs to scale. (Fargate gives no GPU and a fixed
small `/dev/shm` — both already handled: the container is CPU-only and uses the `file_system`
DataLoader strategy.)

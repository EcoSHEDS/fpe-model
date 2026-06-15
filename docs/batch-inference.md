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

Given a station id, model code, and imageset id, the container:

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
   s3://usgs-chs-conte-prod-fpe-models/rank/{station_id}/models/{model_code}/imagesets/{imageset_id}/transform/predictions.csv
   ```
   — the same schema and location the SageMaker `run-transform-predictions.py --imageset-id` path produced.

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
| `--imageset-id` | `IMAGESET_ID` | yes | — | imageset to score |
| `--batch-size` | `BATCH_SIZE` | no | `32` | inference batch size |
| `--num-workers` | `NUM_WORKERS` | no | `4` | `DataLoader` workers (S3 download/decode parallelism) |

Database connection (env only — there is **no** `r/config.yml` and no Secrets Manager API call
in the container code):

| Env var | Description |
|---|---|
| `DB_HOST` | Postgres host |
| `DB_PORT` | Postgres port |
| `DB_NAME` | database name |
| `DB_USER` | database user |
| `DB_PASSWORD` | database password |

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
  -e STATION_ID=29 -e MODEL_CODE=RANK-FLOW-20240709 -e IMAGESET_ID=1234 \
  -e DB_HOST=... -e DB_PORT=5432 -e DB_NAME=... -e DB_USER=... -e DB_PASSWORD=... \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_SESSION_TOKEN=... \
  -e AWS_REGION=us-west-2 \
  fpe-predict
```

You can also run the entrypoint without Docker (needs `requirements.txt` + torch installed):

```sh
STATION_ID=29 MODEL_CODE=RANK-FLOW-20240709 IMAGESET_ID=1234 \
DB_HOST=... DB_PORT=5432 DB_NAME=... DB_USER=... DB_PASSWORD=... \
AWS_PROFILE=conte-prod \
python src/predict-imageset.py
```

## AWS Batch job definition contract

The compute environment, job queue, and IAM are infra-managed; this repo provides only the
container and the code it runs. The job definition must supply:

- **Container image:** the ECR URI pushed above.
- **Command / environment:** `STATION_ID`, `MODEL_CODE`, `IMAGESET_ID` (and optionally
  `BATCH_SIZE`, `NUM_WORKERS`). These can be `containerProperties.command` overrides or
  `environment` entries; the entrypoint accepts either form.
- **Database credentials:** `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`. Inject
  these from Secrets Manager via the job definition `secrets` / `valueFrom` block. The
  `secretsmanager:GetSecretValue` (and any KMS `kms:Decrypt`) permission then lives on the Batch
  **execution role** — infra-side, never baked into the image.
- **Compute:** CPU only — size `vcpus`/`memory` in the job definition for the chosen `--num-workers`
  and image volume. No GPU resource or GPU compute environment is required (Fargate is viable too).

### IAM (Batch *job* role)

The role the container assumes needs:

| Action | Resource | Why |
|---|---|---|
| `s3:GetObject` | `arn:aws:s3:::usgs-chs-conte-prod-fpe-storage/models/*` | download `model.tar.gz` |
| `s3:GetObject` | `arn:aws:s3:::usgs-chs-conte-prod-fpe-storage/*` | download imageset images |
| `s3:GetObject` | `arn:aws:s3:::usgs-chs-conte-prod-fpe-models/rank/*` | read `station.json` / `rank-input.json` |
| `s3:PutObject` | `arn:aws:s3:::usgs-chs-conte-prod-fpe-models/rank/*` | write `predictions.csv` |

Plus network reachability to the Postgres host (security group / subnet). If DB creds come from
Secrets Manager, `secretsmanager:GetSecretValue` goes on the **execution** role (not the job role).

## Notes / tuning

- **Throughput:** S3 download is usually the bottleneck for this small model, so `--num-workers`
  (download concurrency) matters most; `--batch-size` trades memory for CPU forward-pass efficiency.
  Per-image download/decode failures are logged and recorded as `score=NaN` rather than failing the
  whole job.
- **S3 throttling:** the S3 client uses adaptive retries; keep `num_workers` bounded on very
  large imagesets.
- **Timeout:** size the Batch job timeout to expected throughput for tens-of-thousands-image
  imagesets.

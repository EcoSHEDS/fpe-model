# AWS Batch inference container for single-imageset scoring.
# CPU-only: small slim base + CPU torch wheels. This model (ResNet-18) on an
# S3-read-heavy workload is I/O-bound, so a GPU buys little; keeping it CPU keeps the
# image ~4-5x smaller and the infra simple (no GPU compute env / drivers needed).
FROM python:3.9-slim

WORKDIR /opt/fpe

# torch hub cache location, pinned so it's independent of $HOME / the runtime user.
ENV TORCH_HOME=/opt/fpe/.torch

# libgomp1: OpenMP runtime needed by torch / scikit-learn on the slim Debian base.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# the slim base has no torch; install CPU-only torch/torchvision wheels from the PyTorch CPU index,
# then the remaining (non-torch) runtime deps from PyPI.
RUN pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 \
        --index-url https://download.pytorch.org/whl/cpu
COPY requirements-batch.txt .
RUN pip install --no-cache-dir -r requirements-batch.txt

# copy only the modules the inference path needs (no notebooks, training, or run-* scripts)
COPY src/transform.py \
     src/modules.py \
     src/datasets.py \
     src/utils.py \
     src/losses.py \
     src/fpe_inference.py \
     src/fpe_imageset.py \
     src/predict-imageset.py \
     ./

# Pre-cache the torchvision ResNet-18 weights. transform.model_fn builds
# ResNetRankNet(pretrained=True), which downloads these at construct time; the trained
# checkpoint then overwrites them. Caching at build time lets the container run in a Batch
# environment with no internet egress.
RUN python -c "import torchvision; torchvision.models.resnet18(pretrained=True)"

# Batch passes params as args and/or env (STATION_ID/MODEL_CODE/IMAGESET_UUIDS/...);
# DB creds come from env (DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD).
ENTRYPOINT ["python", "predict-imageset.py"]

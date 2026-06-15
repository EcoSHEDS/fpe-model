"""Batched, parity-preserving inference for single-imageset scoring.

Mirrors the SageMaker serving path in transform.py exactly so incremental imageset
scores match what the trained model produced on the station-wide run:

  * model load: ``transform.model_fn`` (ResNetRankNet resnet_size=18 truncate=2,
    transforms/params from model.pth, DataParallel(device_ids=[]) wrap, eval mode)
  * image decode: PIL ``Image.open`` + ``ToTensor`` with NO ``.convert`` -- identical to
    transform.load_from_bytearray (transform.py:46-54)
  * preprocessing: ``model.module.transforms['eval']`` applied per image, identical to
    transform.predict_fn (transform.py:72)
  * forward: ``model.module.forward_single(batch)`` directly, bypassing DataParallel.forward

BatchNorm runs in eval() mode (running stats), so scoring N images in one batch is
numerically identical to the per-image loop.

Inference runs on CPU (this container is CPU-only). A DataLoader with num_workers>0 overlaps
S3 download + image decode/transform with the model's forward pass; S3 download is typically
the bottleneck for this small model. boto3 clients are created lazily per worker process
(clients are not fork/spawn-safe to share).

run_inference sets PyTorch's ``file_system`` tensor-sharing strategy so DataLoader workers
pass tensors via temp files instead of /dev/shm. This is required on AWS Fargate, whose
/dev/shm is a fixed ~64 MB and cannot be enlarged (no linuxParameters.sharedMemorySize); the
default ``file_descriptor`` strategy overflows it and workers die with "Unexpected bus error
... insufficient shared memory (shm)".
"""

import io
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from transform import model_fn

DEFAULT_REGION = os.environ.get("AWS_REGION", "us-west-2")


def _make_s3_client(region=DEFAULT_REGION):
    # adaptive retries to ride out S3 GET throttling on tens of thousands of objects
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=region,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )


class S3ImageDataset(Dataset):
    """Downloads, decodes, and eval-transforms each image; one item per df row.

    __getitem__ returns ``{"idx": i, "image": tensor_or_None}``. A None image (download or
    decode failure) is carried through so the caller can record score=NaN instead of crashing.

    Args:
        df: dataframe with a ``filename`` column (the S3 key under ``storage_bucket``,
            i.e. the url path with no leading slash).
        storage_bucket: S3 bucket holding the images.
        eval_transform: ``model.module.transforms['eval']`` -- applied to the ToTensor output,
            exactly as transform.predict_fn does.
        fetch_bytes: optional ``filename -> bytes`` callable. Defaults to S3 GetObject.
            Injectable so parity tests can feed identical local bytes to both paths.
        region: AWS region for the lazily-created per-worker S3 client.
    """

    def __init__(self, df, storage_bucket, eval_transform, fetch_bytes=None, region=DEFAULT_REGION):
        self.filenames = df["filename"].tolist()
        self.storage_bucket = storage_bucket
        self.eval_transform = eval_transform
        self._fetch_bytes = fetch_bytes
        self.region = region
        self._s3 = None  # created lazily, per worker process (never shared across a fork)

    def __len__(self):
        return len(self.filenames)

    def _fetch(self, filename):
        if self._fetch_bytes is not None:
            return self._fetch_bytes(filename)
        if self._s3 is None:
            self._s3 = _make_s3_client(self.region)
        obj = self._s3.get_object(Bucket=self.storage_bucket, Key=filename)
        return obj["Body"].read()

    def __getitem__(self, index):
        filename = self.filenames[index]
        try:
            raw = self._fetch(filename)
            # PIL Image.open + ToTensor, NO .convert -- mirrors transform.load_from_bytearray
            image = Image.open(io.BytesIO(raw))
            tensor = ToTensor()(image)
            # eval transform applied per-image, mirrors transform.predict_fn
            tensor = self.eval_transform(tensor)
            return {"idx": index, "image": tensor}
        except Exception as e:
            print(f"Error loading image index {index} ({filename}) (score=NaN): {e}")
            return {"idx": index, "image": None}


def _collate(batch):
    """Split a batch into a stacked tensor of decoded images plus their row indices,
    keeping failed (None) rows aside so they can be scored NaN."""
    valid = [b for b in batch if b["image"] is not None]
    failed_idxs = [b["idx"] for b in batch if b["image"] is None]
    if valid:
        images = torch.stack([b["image"] for b in valid])
        valid_idxs = [b["idx"] for b in valid]
    else:
        images = None
        valid_idxs = []
    return images, valid_idxs, failed_idxs


def run_inference(
    df,
    model_dir,
    storage_bucket,
    batch_size=32,
    num_workers=4,
    device=None,
    fetch_bytes=None,
    region=DEFAULT_REGION,
):
    """Score every row of ``df`` and return a copy with a ``score`` column added.

    Reuses transform.model_fn for the model load (parity), runs batched inference via
    ``model.module.forward_single`` under ``torch.no_grad()``, and records score=NaN for any
    image that fails to download/decode. Row order and index of ``df`` are preserved.
    """
    if device is None:
        device = torch.device("cpu")  # CPU-only container
    print(f"run_inference: device={device} batch_size={batch_size} num_workers={num_workers} n={len(df)}")

    model = model_fn(model_dir)
    module = model.module  # the ResNetRankNet; bypasses the unused DataParallel.forward
    module.to(device)
    module.eval()
    eval_transform = module.transforms["eval"]

    # DataLoader workers hand loaded tensors to the main process through shared memory
    # (/dev/shm). On Fargate /dev/shm is a fixed ~64 MB that image batches overflow, and it
    # cannot be enlarged (no linuxParameters.sharedMemorySize on Fargate, and --shm-size is
    # local-docker only) -> "Unexpected bus error ... insufficient shared memory (shm)". The
    # file_system strategy passes tensors via temp files instead, so it works regardless.
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")

    dataset = S3ImageDataset(
        df, storage_bucket, eval_transform, fetch_bytes=fetch_bytes, region=region
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    scores = np.full(len(df), np.nan, dtype=np.float64)
    with torch.no_grad():
        for images, valid_idxs, failed_idxs in loader:
            if images is None:
                continue
            output = module.forward_single(images.to(device))
            # forward_single ends in .squeeze(), which collapses the batch dim when a batch
            # has size 1; reshape(-1) restores a per-row vector in every case.
            preds = output.detach().cpu().numpy().reshape(-1)
            for pos, idx in enumerate(valid_idxs):
                scores[idx] = preds[pos]

    result = df.copy()
    result["score"] = scores
    return result

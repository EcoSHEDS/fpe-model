"""Parity test: batched fpe_inference.run_inference vs the legacy per-image predict_fn path.

Both paths are fed the *same local image bytes* so any score difference is attributable to
batching alone. BatchNorm in eval() mode uses running stats, so the expected difference is ~0;
the acceptance threshold is max abs diff <= 1e-5.

This test needs torch + a real trained model + a few images, none of which live in the repo.
It SKIPS unless all of the following are available:
  * torch / torchvision importable
  * env FPE_TEST_MODEL_DIR -> a directory containing model.pth (extracted model.tar.gz)
  * env FPE_TEST_IMAGES_DIR -> a directory containing a few image files the model can score

Run:
  FPE_TEST_MODEL_DIR=/path/to/model_dir \
  FPE_TEST_IMAGES_DIR=/path/to/images \
  pytest tests/test_inference_parity.py -q -s
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torchvision", reason="torchvision not installed")

MODEL_DIR = os.environ.get("FPE_TEST_MODEL_DIR")
IMAGES_DIR = os.environ.get("FPE_TEST_IMAGES_DIR")

_IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def _collect_images(images_dir, limit=5):
    files = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(files)[:limit]


@pytest.mark.skipif(
    not MODEL_DIR or not os.path.exists(os.path.join(MODEL_DIR or "", "model.pth")),
    reason="set FPE_TEST_MODEL_DIR to a dir containing model.pth",
)
@pytest.mark.skipif(
    not IMAGES_DIR or not os.path.isdir(IMAGES_DIR or ""),
    reason="set FPE_TEST_IMAGES_DIR to a dir containing test images",
)
def test_batched_matches_legacy_predict_fn():
    import transform
    import fpe_inference

    image_paths = _collect_images(IMAGES_DIR)
    assert len(image_paths) >= 1, f"no images found in {IMAGES_DIR}"

    # key images by basename; feed both paths the exact same bytes from this dict
    local_bytes = {}
    for p in image_paths:
        with open(p, "rb") as f:
            local_bytes[os.path.basename(p)] = f.read()
    filenames = list(local_bytes.keys())

    df = pd.DataFrame({
        "split": "test-out",
        "image_id": range(len(filenames)),
        "timestamp": pd.Timestamp("2024-07-01T12:00:00Z"),
        "filename": filenames,
        "url": filenames,
        "value": pd.NA,
    })

    # batched path -- inject the local byte source; num_workers=0 so the closure needn't pickle.
    # batch_size=2 over an odd count forces a trailing batch of size 1 (the .squeeze edge case).
    result = fpe_inference.run_inference(
        df,
        MODEL_DIR,
        storage_bucket="unused-in-test",
        batch_size=2,
        num_workers=0,
        fetch_bytes=lambda fn: local_bytes[fn],
    )
    batched_scores = result["score"].to_numpy()
    assert not np.isnan(batched_scores).any(), "all fixture images should score (none NaN)"

    # legacy per-image path -- transform.load_from_bytearray + predict_fn, same model + bytes
    model = transform.model_fn(MODEL_DIR)
    model.eval()
    legacy_scores = np.empty(len(filenames), dtype=np.float64)
    with torch.no_grad():
        for i, fn in enumerate(filenames):
            obj = transform.load_from_bytearray(local_bytes[fn])
            legacy_scores[i] = transform.predict_fn(obj, model)["score"]

    max_abs_diff = float(np.max(np.abs(batched_scores - legacy_scores)))
    print(f"\nparity: n={len(filenames)} max_abs_diff={max_abs_diff:.3e}")
    print(f"  batched: {np.round(batched_scores, 6)}")
    print(f"  legacy : {np.round(legacy_scores, 6)}")
    assert max_abs_diff <= 1e-5, f"batched vs legacy diverged: max abs diff {max_abs_diff:.3e} > 1e-5"

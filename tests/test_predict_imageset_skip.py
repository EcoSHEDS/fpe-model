import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fpe_imageset  # noqa: E402


pytest.importorskip("boto3")
pytest.importorskip("botocore")


class _Body:
    def __init__(self, payload):
        self.payload = payload.encode("utf-8")

    def read(self):
        return self.payload


class _FakeS3:
    def __init__(self):
        self.puts = []

    def get_object(self, Bucket, Key):
        if Key.endswith("/input/station.json"):
            return {"Body": _Body(json.dumps({"timezone": "America/New_York"}))}
        if Key.endswith("/input/rank-input.json"):
            return {"Body": _Body(json.dumps({"args": {"options": {}}}))}
        raise AssertionError(f"unexpected get_object key: {Key}")

    def put_object(self, **kwargs):
        self.puts.append(kwargs)


class _FakeConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _load_predict_module():
    path = Path(__file__).parents[1] / "src" / "predict-imageset.py"
    spec = importlib.util.spec_from_file_location("predict_imageset_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_common(monkeypatch, module, fake_s3):
    fake_conn = _FakeConn()
    monkeypatch.setattr(module, "_make_s3", lambda: fake_s3)
    monkeypatch.setattr(module, "get_db_config", lambda: {})
    monkeypatch.setattr(module, "connect_db", lambda db: fake_conn)
    monkeypatch.setattr(
        module, "fetch_model_uuid", lambda conn, station_id, code: "model-uuid"
    )
    return fake_conn


def test_predict_imagesets_skips_all_empty_filtered_imagesets(monkeypatch, capsys):
    module = _load_predict_module()
    fake_s3 = _FakeS3()
    fake_conn = _patch_common(monkeypatch, module, fake_s3)

    def build_dataframe(conn, imageset_uuid, station_id, timezone, filters):
        raise fpe_imageset.EmptyImagesetAfterFilter("no images remain after filtering")

    monkeypatch.setattr(module, "build_imageset_dataframe", build_dataframe)
    monkeypatch.setattr(
        module,
        "_download_and_extract_model",
        lambda *args, **kwargs: pytest.fail("model should not download when all imagesets skip"),
    )

    out = module.predict_imagesets(
        "42", "RANK-X", ["u1", "u2"], batch_size=32, num_workers=4
    )

    assert out == []
    assert fake_conn.closed
    assert fake_s3.puts == []
    stdout = capsys.readouterr().out
    assert "done: 0 of 2 imagesets scored; skipped=2 failed=0" in stdout
    assert "skipped imagesets: u1, u2" in stdout


def test_predict_imagesets_scores_non_empty_and_reports_skips(monkeypatch, capsys):
    module = _load_predict_module()
    fake_s3 = _FakeS3()
    _patch_common(monkeypatch, module, fake_s3)

    def build_dataframe(conn, imageset_uuid, station_id, timezone, filters):
        if imageset_uuid == "skip-me":
            raise fpe_imageset.EmptyImagesetAfterFilter("no images remain after filtering")
        return pd.DataFrame(
            [
                {
                    "image_id": 1,
                    "timestamp": "2024-07-01T11:00:00Z",
                    "filename": "img/a.jpg",
                    "url": "https://b.s3.amazonaws.com/img/a.jpg",
                }
            ]
        )

    def run_inference(df, model_dir, storage_bucket, batch_size, num_workers, region):
        result = df.copy()
        result["score"] = [0.75]
        return result

    monkeypatch.setattr(module, "build_imageset_dataframe", build_dataframe)
    monkeypatch.setattr(module, "_download_and_extract_model", lambda *args: "/tmp/model")
    monkeypatch.setitem(
        sys.modules, "fpe_inference", types.SimpleNamespace(run_inference=run_inference)
    )

    out = module.predict_imagesets(
        "42", "RANK-X", ["skip-me", "score-me"], batch_size=32, num_workers=4
    )

    assert out == ["rank/42/models/RANK-X/transform/imagesets/score-me/predictions.csv"]
    assert len(fake_s3.puts) == 1
    assert "score-me/predictions.csv" in fake_s3.puts[0]["Key"]
    assert "0.75" in fake_s3.puts[0]["Body"]
    stdout = capsys.readouterr().out
    assert "done: 1 of 2 imagesets scored; skipped=1 failed=0" in stdout
    assert "skipped imagesets: skip-me" in stdout

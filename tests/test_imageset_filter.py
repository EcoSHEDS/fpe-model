"""Offline tests for fpe_imageset.build_imageset_dataframe.

No database or network: fetch_imageset_station / fetch_imageset_images are monkeypatched
with synthetic rows. Verifies the daytime/seasonal filter, timezone conversion, output
schema, sort order, and the stats out-param.

Run: pytest tests/test_imageset_filter.py
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fpe_imageset  # noqa: E402


# America/New_York is UTC-4 in July (DST). 11:00, 15:00, 23:00 UTC -> 07:00, 11:00, 19:00 local.
SYNTH_IMAGES = pd.DataFrame(
    [
        # local 07:00 July -> inside hour [7,18] and month [1,12] -> KEEP
        {"image_id": 1, "imageset_id": 9, "timestamp": "2024-07-01T11:00:00Z",
         "url": "https://b.s3.amazonaws.com/img/a.jpg"},
        # local 11:00 July -> KEEP
        {"image_id": 2, "imageset_id": 9, "timestamp": "2024-07-01T15:00:00Z",
         "url": "https://b.s3.amazonaws.com/img/b.jpg"},
        # local 19:00 July -> hour 19 > 18 -> DROP
        {"image_id": 3, "imageset_id": 9, "timestamp": "2024-07-01T23:00:00Z",
         "url": "https://b.s3.amazonaws.com/img/c.jpg"},
    ]
)


def _patch(monkeypatch, station_id=42, images=None):
    monkeypatch.setattr(fpe_imageset, "fetch_imageset_station", lambda conn, iid: station_id)
    df = SYNTH_IMAGES if images is None else images
    monkeypatch.setattr(fpe_imageset, "fetch_imageset_images", lambda conn, iid: df.copy())


def test_filter_schema_and_order(monkeypatch):
    _patch(monkeypatch)
    stats = {}
    out = fpe_imageset.build_imageset_dataframe(
        conn=None, imageset_id=9, station_id=42,
        timezone="America/New_York", filters=fpe_imageset.DEFAULT_FILTERS, stats=stats,
    )
    # exact column schema / order that images.csv requires
    assert list(out.columns) == ["split", "image_id", "timestamp", "filename", "url", "value"]
    # 19:00-local row dropped by the hour filter; 2 of 3 remain
    assert list(out["image_id"]) == [1, 2]
    assert stats == {"total": 3, "filtered": 2}
    # constants written for every imageset row
    assert (out["split"] == "test-out").all()
    assert out["value"].isna().all()
    # filename = url path without leading slash
    assert list(out["filename"]) == ["img/a.jpg", "img/b.jpg"]
    # sorted by timestamp ascending
    assert out["timestamp"].is_monotonic_increasing


def test_station_mismatch_raises(monkeypatch):
    _patch(monkeypatch, station_id=99)
    with pytest.raises(Exception, match="belongs to station 99"):
        fpe_imageset.build_imageset_dataframe(
            conn=None, imageset_id=9, station_id=42,
            timezone="America/New_York", filters=fpe_imageset.DEFAULT_FILTERS,
        )


def test_imageset_not_found_raises(monkeypatch):
    monkeypatch.setattr(fpe_imageset, "fetch_imageset_station", lambda conn, iid: None)
    with pytest.raises(Exception, match="imageset not found"):
        fpe_imageset.build_imageset_dataframe(
            conn=None, imageset_id=9, station_id=42,
            timezone="America/New_York", filters=fpe_imageset.DEFAULT_FILTERS,
        )


def test_empty_after_filter_raises(monkeypatch):
    # single row at local 19:00 (dropped) -> nothing survives the filter
    only_dropped = SYNTH_IMAGES.iloc[[2]].reset_index(drop=True)
    _patch(monkeypatch, images=only_dropped)
    with pytest.raises(Exception, match="no images remain"):
        fpe_imageset.build_imageset_dataframe(
            conn=None, imageset_id=9, station_id=42,
            timezone="America/New_York", filters=fpe_imageset.DEFAULT_FILTERS,
        )


def test_get_db_config_missing_env(monkeypatch):
    for var in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="DB_HOST"):
        fpe_imageset.get_db_config()


def test_get_db_config_reads_env(monkeypatch):
    monkeypatch.setenv("DB_HOST", "h")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "n")
    monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "p")
    cfg = fpe_imageset.get_db_config()
    assert cfg == {"host": "h", "port": "5432", "dbname": "n", "user": "u", "password": "p"}


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed = None

    def execute(self, sql, params=None):
        self.executed = (sql, params)

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, rows):
        self._cur = _FakeCursor(rows)

    def cursor(self):
        return self._cur


def test_fetch_model_uuid_exactly_one():
    conn = _FakeConn([("00abdca8-47df-4afe-a94a-84a7238d480d",)])
    uuid = fpe_imageset.fetch_model_uuid(conn, station_id=29, model_code="RANK-FLOW-X")
    assert uuid == "00abdca8-47df-4afe-a94a-84a7238d480d"
    sql, params = conn._cur.executed
    assert "SELECT uuid FROM models WHERE station_id = %s AND code = %s" in sql
    assert params == (29, "RANK-FLOW-X")


def test_fetch_model_uuid_not_found():
    with pytest.raises(Exception, match="no model found"):
        fpe_imageset.fetch_model_uuid(_FakeConn([]), 29, "RANK-FLOW-X")


def test_fetch_model_uuid_ambiguous():
    with pytest.raises(Exception, match="ambiguous"):
        fpe_imageset.fetch_model_uuid(_FakeConn([("u1",), ("u2",)]), 29, "RANK-FLOW-X")

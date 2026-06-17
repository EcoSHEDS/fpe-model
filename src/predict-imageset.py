#!/usr/bin/env python
"""Self-contained AWS Batch entrypoint: score one or more imagesets, write drop-in predictions.csv files.

Replaces the SageMaker transform -> merge -> predictions chain for incremental imageset
scoring. Given a station id, model code, and one or more imageset UUIDs, this:
  1. reads the model's station.json + rank-input.json from the model bucket (timezone, filters),
  2. looks up the model UUID in the DB and pulls models/{uuid}/model.tar.gz from the storage bucket,
  3. builds the filtered DONE-image dataframe from Postgres for each imageset (fpe_imageset; looked up by uuid),
  4. runs batched, parity-preserving inference on CPU (fpe_inference), reusing the single downloaded model,
  5. writes predictions.csv (image_id,timestamp,filename,url,score) to each imageset's
     transform key on the model bucket -- identical schema/location to the SageMaker path.

The model config and model artifact are fetched once and reused across every imageset. Each
imageset is scored independently: a failure on one is logged and the rest still run, but the
job exits non-zero if any imageset failed so Batch marks it FAILED. Imagesets that have
DONE images but zero rows after the model's filters are skipped and reported at the end.

Params come from CLI args or env vars (Batch may pass either). DB credentials come from
FPE_DB_SECRET (a Secrets Manager secret name) if set, else the discrete DB_HOST/DB_PORT/
DB_NAME/DB_USER/DB_PASSWORD vars. The S3 buckets default to the prod buckets but can be
overridden via FPE_S3_STORAGE_BUCKET / FPE_S3_MODEL_BUCKET.

Usage:
  python src/predict-imageset.py --station-id S --model-code M --imageset-uuids U1,U2,U3 \
      [--batch-size 32] [--num-workers 4]
Env equivalents: STATION_ID, MODEL_CODE, IMAGESET_UUIDS (or IMAGESET_UUID), BATCH_SIZE, NUM_WORKERS.
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
from io import StringIO

import boto3
from botocore.config import Config

from fpe_imageset import (
    EmptyImagesetAfterFilter,
    build_imageset_dataframe,
    connect_db,
    fetch_model_uuid,
    get_db_config,
    resolve_filters,
)

# fpe_inference (and torch) is imported lazily inside predict_imageset() so that --help and
# argument validation work without torch, and bad args fail fast before the heavy import.

# buckets / region: default to the prod buckets (as in the SageMaker scripts), overridable by env
MODEL_BUCKET = os.environ.get("FPE_S3_MODEL_BUCKET", "usgs-chs-conte-prod-fpe-models")
STORAGE_BUCKET = os.environ.get("FPE_S3_STORAGE_BUCKET", "usgs-chs-conte-prod-fpe-storage")
REGION = os.environ.get("AWS_REGION", "us-west-2")


def _make_s3():
    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )


def _fetch_json(s3, bucket, key):
    print(f"fetching: s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def _download_and_extract_model(s3, uuid, workdir):
    """Download models/{uuid}/model.tar.gz from the storage bucket and extract it.

    Returns the directory that contains model.pth (the model_dir transform.model_fn expects).
    """
    key = f"models/{uuid}/model.tar.gz"
    local_tar = os.path.join(workdir, "model.tar.gz")
    print(f"downloading model: s3://{STORAGE_BUCKET}/{key}")
    s3.download_file(STORAGE_BUCKET, key, local_tar)

    model_dir = os.path.join(workdir, "model")
    os.makedirs(model_dir, exist_ok=True)
    print(f"extracting model -> {model_dir}")
    with tarfile.open(local_tar) as tar:
        tar.extractall(model_dir)

    model_pth = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_pth):
        raise Exception(f"model.pth not found in extracted artifact ({key})")
    return model_dir


def predict_imagesets(station_id, model_code, imageset_uuids, batch_size, num_workers):
    print(
        f"predict_imagesets: station={station_id} model={model_code} "
        f"imagesets={','.join(imageset_uuids)} ({len(imageset_uuids)}) "
        f"batch_size={batch_size} num_workers={num_workers}"
    )
    s3 = _make_s3()

    # model config (timezone + filters) from the model bucket input dir -- shared by all imagesets
    model_key = f"rank/{station_id}/models/{model_code}"
    station = _fetch_json(s3, MODEL_BUCKET, f"{model_key}/input/station.json")
    timezone = station["timezone"]
    rank_input = _fetch_json(s3, MODEL_BUCKET, f"{model_key}/input/rank-input.json")
    filters = resolve_filters(rank_input)
    print(f"timezone: {timezone}")
    print(f"filters: {filters}")

    failures = []
    skipped = []

    # DB: resolve the model UUID and build the filtered image dataframe for each imageset
    conn = connect_db(get_db_config())
    try:
        uuid = fetch_model_uuid(conn, station_id, model_code)
        print(f"model uuid: {uuid}")
        dataframes = {}
        for imageset_uuid in imageset_uuids:
            try:
                dataframes[imageset_uuid] = build_imageset_dataframe(
                    conn, imageset_uuid, station_id, timezone, filters
                )
            except EmptyImagesetAfterFilter as e:
                print(f"SKIP imageset {imageset_uuid}: {e}")
                skipped.append(imageset_uuid)
            except Exception as e:
                print(f"ERROR building dataframe for imageset {imageset_uuid}: {e}", file=sys.stderr)
                failures.append(imageset_uuid)
    finally:
        conn.close()

    if not dataframes:
        print(
            f"done: 0 of {len(imageset_uuids)} imagesets scored; "
            f"skipped={len(skipped)} failed={len(failures)}"
        )
        if skipped:
            print(f"skipped imagesets: {', '.join(skipped)}")
        if failures:
            raise Exception(
                f"{len(failures)} of {len(imageset_uuids)} imagesets failed: "
                f"{', '.join(failures)}"
            )
        return []

    # download + extract the model once, then run batched inference per imageset under a temp workdir
    import fpe_inference

    out_keys = []
    with tempfile.TemporaryDirectory() as workdir:
        model_dir = _download_and_extract_model(s3, uuid, workdir)
        for imageset_uuid, df in dataframes.items():
            try:
                result = fpe_inference.run_inference(
                    df,
                    model_dir,
                    STORAGE_BUCKET,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    region=REGION,
                )

                # write the drop-in predictions.csv to the imageset transform key on the model bucket
                out_key = f"{model_key}/transform/imagesets/{imageset_uuid}/predictions.csv"
                buf = StringIO()
                result.to_csv(buf, index=False)
                s3.put_object(Bucket=MODEL_BUCKET, Key=out_key, Body=buf.getvalue())
                n_scored = int(result["score"].notna().sum())
                print(
                    f"wrote s3://{MODEL_BUCKET}/{out_key} "
                    f"(rows={len(result)}, scored={n_scored}, nan={len(result) - n_scored})"
                )
                out_keys.append(out_key)
            except Exception as e:
                print(f"ERROR scoring imageset {imageset_uuid}: {e}", file=sys.stderr)
                failures.append(imageset_uuid)

    print(
        f"done: {len(out_keys)} of {len(imageset_uuids)} imagesets scored; "
        f"skipped={len(skipped)} failed={len(failures)}"
    )
    if skipped:
        print(f"skipped imagesets: {', '.join(skipped)}")
    if failures:
        raise Exception(
            f"{len(failures)} of {len(imageset_uuids)} imagesets failed: {', '.join(failures)}"
        )
    return out_keys


def _resolve(cli_value, env_name, default=None, required=False, cast=str):
    """CLI arg wins, else env var, else default. Raise if required and unresolved."""
    value = cli_value if cli_value is not None else os.environ.get(env_name)
    if value is None or value == "":
        if required:
            raise RuntimeError(
                f"missing required parameter: pass --{env_name.lower().replace('_', '-')} "
                f"or set {env_name}"
            )
        value = default
    return cast(value) if value is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-id", type=str, default=None)
    parser.add_argument("--model-code", type=str, default=None)
    # --imageset-uuid is kept as a backward-compatible alias; either flag accepts a
    # single UUID or a comma-separated list.
    parser.add_argument(
        "--imageset-uuids",
        "--imageset-uuid",
        dest="imageset_uuids",
        type=str,
        default=None,
        help="one or more imageset UUIDs, comma-separated",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    station_id = _resolve(args.station_id, "STATION_ID", required=True)
    model_code = _resolve(args.model_code, "MODEL_CODE", required=True)
    # CLI flag wins; else IMAGESET_UUIDS, else the legacy IMAGESET_UUID env var.
    imageset_uuids_raw = (
        args.imageset_uuids
        or os.environ.get("IMAGESET_UUIDS")
        or os.environ.get("IMAGESET_UUID")
    )
    if not imageset_uuids_raw:
        raise RuntimeError(
            "missing required parameter: pass --imageset-uuids or set IMAGESET_UUIDS"
        )
    imageset_uuids = [u.strip() for u in imageset_uuids_raw.split(",") if u.strip()]
    if not imageset_uuids:
        raise RuntimeError("no imageset UUIDs provided")
    batch_size = _resolve(args.batch_size, "BATCH_SIZE", default=32, cast=int)
    num_workers = _resolve(args.num_workers, "NUM_WORKERS", default=4, cast=int)

    predict_imagesets(station_id, model_code, imageset_uuids, batch_size, num_workers)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # non-zero exit so AWS Batch marks the job FAILED with a clear message
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

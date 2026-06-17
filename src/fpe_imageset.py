"""Importable core for single-imageset scoring: DB connection, image fetch, and
the daytime/seasonal filtering that selects which images a trained model scores.

Used by the containerized AWS Batch entrypoint (``predict-imageset.py``); the
daytime/seasonal filtering mirrors ``rank-input.R`` so an imageset is scored on the
same images the model was built for.

Credentials come **only** from environment variables (no ``r/config.yml``, no
Secrets Manager API call). In AWS Batch these vars are typically injected from
Secrets Manager via the job definition ``secrets``/``valueFrom`` block (infra-side).
"""

import json
import os
from urllib.parse import urlparse

import pandas as pd
import psycopg2

# rank-input.R filter defaults (used when a key is missing from rank-input.json).
# Only the daytime (hour) and seasonal (month) filters are reused for an imageset: they
# define which images the model is meant to score. The model's images_start/images_end are
# deliberately NOT applied -- they bound the model's *training period* (images_end is the
# training date), so applying them would drop every image in a newly-uploaded imageset.
DEFAULT_FILTERS = {
    "min_hour": 7,
    "max_hour": 18,
    "min_month": 1,
    "max_month": 12,
}

# connection params read from these environment variables (no config.yml)
_DB_ENV_VARS = {
    "host": "DB_HOST",
    "port": "DB_PORT",
    "dbname": "DB_NAME",
    "user": "DB_USER",
    "password": "DB_PASSWORD",
}


def get_url_path(url):
    # matches utils.get_url_path / httr::parse_url(url)$path in R: no leading slash
    return urlparse(url).path[1:]


def get_db_config():
    """Build the DB connection params, preferring a Secrets Manager secret.

    If ``FPE_DB_SECRET`` is set, fetch that secret (a JSON document) from Secrets Manager
    and map its fields to connection params (matches the existing FPE batch jobs and the
    Batch job role's SecretsManagerReadWrite). Otherwise fall back to the discrete
    DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD environment variables.
    """
    secret_name = os.environ.get("FPE_DB_SECRET")
    if secret_name:
        return _db_config_from_secret(secret_name)
    return _db_config_from_env()


def _db_config_from_env():
    config = {}
    missing = []
    for key, env_var in _DB_ENV_VARS.items():
        value = os.environ.get(env_var)
        if value is None or value == "":
            missing.append(env_var)
        else:
            config[key] = value
    if missing:
        raise RuntimeError(
            "missing required database environment variable(s): "
            + ", ".join(missing)
            + " (set DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, or set FPE_DB_SECRET)"
        )
    return config


def _map_secret_to_db_config(secret):
    """Map a DB-credentials secret dict to psycopg2 connection params.

    Accepts both the R `config` key style (database/user) and the RDS-managed-secret
    style (dbname/username) for the db name and user fields. The db name falls back to
    "postgres" when the secret omits it (RDS-managed secrets often carry no dbname).
    """
    _REQUIRED = object()

    def pick(*keys, default=_REQUIRED):
        for k in keys:
            value = secret.get(k)
            if value is not None and value != "":
                return value
        if default is not _REQUIRED:
            return default
        raise RuntimeError(
            f"FPE_DB_SECRET is missing a required field (one of {keys})"
        )

    return {
        "host": pick("host"),
        "port": pick("port"),
        "dbname": pick("dbname", "database", default="postgres"),
        "user": pick("user", "username"),
        "password": pick("password"),
    }


def _db_config_from_secret(secret_name):
    import boto3  # lazy: only needed when a secret is configured

    region = os.environ.get("AWS_REGION", "us-west-2")
    client = boto3.client("secretsmanager", region_name=region)
    secret = json.loads(client.get_secret_value(SecretId=secret_name)["SecretString"])
    return _map_secret_to_db_config(secret)


def connect_db(db):
    conn = psycopg2.connect(
        host=db["host"],
        port=db["port"],
        dbname=db["dbname"],
        user=db["user"],
        password=db["password"],
    )
    # match R (Sys.setenv(TZ="GMT")): return timestamps in UTC regardless of column type
    with conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'UTC'")
    return conn


def resolve_filters(rank_input):
    """Merge the model's rank-input.json options over DEFAULT_FILTERS.

    Only the daytime/seasonal keys in DEFAULT_FILTERS are pulled through; any other
    options in rank-input.json are ignored.
    """
    options = rank_input.get("args", {}).get("options", {})
    return {k: options.get(k, v) for k, v in DEFAULT_FILTERS.items()}


def fetch_model_uuid(conn, station_id, model_code):
    """Look up a deployed model's UUID by station + code in the ``models`` table.

    The deployed package lives at ``s3://<storage_bucket>/models/{uuid}/model.tar.gz``
    (see batch-deploy.sh / rank-model-db.R). The query must return exactly one row:
    0 -> not found; >1 -> ambiguous (operator must disambiguate).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT uuid FROM models WHERE station_id = %s AND code = %s",
            (station_id, model_code),
        )
        rows = cur.fetchall()
    if len(rows) == 0:
        raise Exception(
            f"no model found in models table for station {station_id}, code {model_code}"
        )
    if len(rows) > 1:
        raise Exception(
            f"ambiguous model lookup: {len(rows)} rows for station {station_id}, "
            f"code {model_code}; operator must disambiguate"
        )
    return str(rows[0][0])


def fetch_imageset_station(conn, imageset_uuid):
    with conn.cursor() as cur:
        cur.execute("SELECT station_id FROM imagesets WHERE uuid = %s", (imageset_uuid,))
        row = cur.fetchone()
    return row[0] if row else None


def fetch_imageset_images(conn, imageset_uuid):
    sql = """
        SELECT i.id AS image_id, i.imageset_id, i.timestamp, i.full_url AS url
        FROM images i
        JOIN imagesets iset ON iset.id = i.imageset_id
        WHERE iset.uuid = %s
        AND NOT image_has_pii(pii_person, pii_vehicle, pii_on, pii_off)
        AND i.status = 'DONE'
    """
    return pd.read_sql_query(sql, conn, params=(imageset_uuid,))


def build_imageset_dataframe(conn, imageset_uuid, station_id, timezone, filters, stats=None):
    """Fetch + filter a single imageset's DONE images into the transform-input schema.

    Validates the imageset belongs to ``station_id``, fetches its DONE images, applies
    the model's daytime/seasonal filters in the station's local timezone, and returns a
    dataframe with columns ``split, image_id, timestamp, filename, url, value`` sorted by
    timestamp -- the transform-input schema consumed by the inference step.

    The imageset is identified by its ``uuid`` (the public identifier used in image storage
    paths, ``imagesets/{uuid}/...``), not the integer primary key.

    If ``stats`` is a dict, it is populated with ``total`` (DONE images before filtering)
    and ``filtered`` (rows returned) so callers can log counts without re-querying.

    Raises on: imageset not found, imageset/station mismatch, no DONE images, or no
    images remaining after the filter.
    """
    imageset_station_id = fetch_imageset_station(conn, imageset_uuid)
    if imageset_station_id is None:
        raise Exception(f"imageset not found (uuid={imageset_uuid})")
    if str(imageset_station_id) != str(station_id):
        raise Exception(
            f"imageset {imageset_uuid} belongs to station {imageset_station_id}, "
            f"not station {station_id}"
        )

    print(f"fetching: images (imageset_uuid={imageset_uuid})")
    images = fetch_imageset_images(conn, imageset_uuid)

    n_total = len(images)
    if stats is not None:
        stats["total"] = n_total
    print(f"# images (DONE): {n_total}")
    if n_total == 0:
        raise Exception(
            f"no DONE images found for imageset {imageset_uuid} "
            f"(still processing, or wrong imageset uuid)"
        )

    # derive filename from url (no leading slash; manifest prefixes s3://...-storage/)
    images["filename"] = images["url"].apply(get_url_path)

    # apply the model's daytime/seasonal filters in the station's local timezone
    # (matches the hour/month filtering in rank-input.R)
    ts_local = pd.to_datetime(images["timestamp"], utc=True).dt.tz_convert(timezone)
    images["timestamp"] = ts_local
    mask = (
        (ts_local.dt.hour >= filters["min_hour"])
        & (ts_local.dt.hour <= filters["max_hour"])
        & (ts_local.dt.month >= filters["min_month"])
        & (ts_local.dt.month <= filters["max_month"])
    )
    images = images.loc[mask].copy()
    n_filtered = len(images)
    if stats is not None:
        stats["filtered"] = n_filtered
    pct = (n_filtered / n_total) if n_total else 0
    print(f"# images (after filter): {n_filtered} ({pct:.0%} of {n_total})")
    if n_filtered == 0:
        raise Exception(
            "no images remain after applying the model's filters "
            f"({filters}); nothing to predict"
        )

    out = images.sort_values("timestamp")[
        ["image_id", "timestamp", "filename", "url", "value"]
    ]
    return out

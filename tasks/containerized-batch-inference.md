# Task: Containerized AWS Batch inference for a single imageset

## Progress checklist (for the implementing agent)

- [ ] **Task 1** — Extract importable imageset DB/filter core → `src/fpe_imageset.py`; refactor `imageset-input.py` to wrap it
- [ ] **Task 2** — Batched inference module `src/fpe_inference.py` + parity test
- [ ] **Checkpoint: Foundation** — both modules import; parity test passes; `imageset-input.py` output unchanged
- [ ] **Task 3** — Self-contained entrypoint `src/predict-imageset.py`
- [ ] **Task 4** — Resolve model artifact via DB `models` table (station+code → uuid → storage bucket `model.tar.gz`)
- [ ] **Checkpoint: Core** — local entrypoint run produces correct `predictions.csv` on S3 for a real imageset
- [ ] **Task 5** — Dockerfile + `.dockerignore` + `requirements-batch.txt`
- [ ] **Task 6** — Usage docs + Batch job/IAM contract
- [ ] **Checkpoint: Complete** — scores a tens-of-thousands-image imageset end-to-end; parity confirmed

---

## Context

Earlier we added a SageMaker Batch Transform path for scoring a single imageset (`--imageset-id` on
the existing `run-transform*.py` scripts + `src/imageset-input.py`). That path is fine for occasional
small batches, but the workload is now **many** jobs, each imageset up to **tens of thousands of
images**. SageMaker Batch Transform is a poor fit at that scale — `SingleRecord` strategy = one HTTP
request + one forward pass per image, CPU instance, a fan-out of tiny S3 objects, and a Lambda merge
step. It is neither throughput- nor cost-efficient, and the multi-step async chain adds latency.

This builds a **self-contained Docker container** that runs in AWS Batch and replaces the
transform→merge→predictions chain for imageset scoring. Given station/model/imageset IDs, the
container queries Postgres for the imageset's images, pulls the trained model from S3, runs
**batched** inference on CPU, and writes a drop-in `predictions.csv` to
S3 — one job, one output, no Lambda. The compute environment, job queue, and IAM are assumed to
exist; **scope here is the container and the code it runs**, not the Batch infra.

**Confirmed decisions:** self-contained (container does the DB fetch, DB creds from **environment
variables only** — no `r/config.yml`); **CPU-only image** (slim base + CPU torch wheels — the GPU
option was dropped: this small model on an S3-read-heavy workload is I/O-bound, and the original
SageMaker path already scored on CPU); images read via in-container boto3 parallel downloads.

## Architecture Decisions

- **Score parity is the non-negotiable constraint.** Incremental imageset scores must match what the
  trained model produces on the original station-wide run. Reuse the exact model-load and
  preprocessing path: `model_fn` from [transform.py](../src/transform.py) (constructs
  `ResNetRankNet(resnet_size=18, truncate=2)`, loads `transforms`/`params` from `model.pth`,
  `DataParallel` wrap), and decode images as **PIL `Image.open` + `ToTensor`** with **no `.convert`**
  to mirror [transform.py:46-54](../src/transform.py#L46-L54). BatchNorm in `eval()` mode uses running
  stats, so batching N images is numerically identical to the current per-image loop.
- **Inference calls `model.module.forward_single(batch)` directly** on CPU, bypassing
  `DataParallel.forward` (the existing `device_ids=[]` wrap only affects the unused
  `DataParallel.forward`, so calling `.module` directly is the clean path).
- **Reuse via importable modules.** The current entrypoint scripts are hyphenated
  (`run-transform.py`, `imageset-input.py`) and can't be imported. Extract the reusable DB/filter and
  inference logic into underscore modules (`fpe_imageset.py`, `fpe_inference.py`) so the container and
  the existing CLI share one source of truth. `transform.py`, `modules.py`, `datasets.py`, `utils.py`
  are already importable and reused as-is.
- **Drop-in output contract.** Write `predictions.csv` with columns
  `split,image_id,timestamp,filename,url,value,score` to
  `s3://usgs-chs-conte-prod-fpe-models/rank/{station_id}/models/{model_code}/imagesets/{imageset_id}/transform/predictions.csv`
  — identical schema/location to what `run-transform-predictions.py --imageset-id` would produce.
- **DataLoader-driven I/O overlap.** A `Dataset` whose `__getitem__` downloads (boto3 GetObject),
  decodes, and applies eval transforms, run through a `DataLoader(num_workers=K)`, overlaps S3
  download + decode with the model's forward pass (S3 download is the likely bottleneck). boto3
  clients are created per-worker (fork-unsafe to share).

## Buckets / constants (reuse existing)
- model bucket `usgs-chs-conte-prod-fpe-models`, storage bucket `usgs-chs-conte-prod-fpe-storage`,
  region `us-west-2`. Image S3 key = `storage_bucket` + `/` + `filename` (filename = url path, no
  leading slash, via `utils.get_url_path`). Model artifact (deployed) at the **storage** bucket
  `models/{uuid}/model.tar.gz`, where `uuid` is looked up in the DB `models` table by station_id +
  code; config JSON at the **model** bucket `rank/{station}/models/{model}/input/{station.json,rank-input.json}`.

## Task List

### Phase 1: Shared inference + imageset core (foundation)

#### Task 1: Extract importable imageset DB/filter core
**Description:** Move the DB-connect, station-validation, image-fetch, and filter logic out of the
hyphenated `imageset-input.py` into a new importable module `src/fpe_imageset.py`, and add a
Secrets-Manager-aware credential loader. Refactor `imageset-input.py` to thin-wrap it (no behavior
change for the existing CLI).
**Acceptance criteria:**
- [ ] `src/fpe_imageset.py` exposes `get_db_config()`, `connect_db()`, `fetch_imageset_station()`, `fetch_imageset_images()`, and `build_imageset_dataframe(conn, imageset_id, station_id, timezone, filters) -> df` (df has `split,image_id,timestamp,filename,url,value`).
- [ ] `get_db_config()` reads connection params **only from environment variables** (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`); **no `r/config.yml` and no Secrets Manager API call** in code. Raise a clear error if any required var is missing. (In Batch, these vars can be populated from Secrets Manager via the job definition `secrets`/`valueFrom` — infra-side.)
- [ ] `imageset-input.py` imports from `fpe_imageset` and produces byte-identical `images.csv` output to before. Its `--config`/`r/config.yml` credential path is removed in favor of the env vars above; the produced CSV is unchanged.
**Verification:** `python -m py_compile`; run `imageset-input.py` (or a unit harness) against a tiny fixture and diff the resulting `images.csv` columns/rows.
**Dependencies:** None
**Files:** `src/fpe_imageset.py` (new), `src/imageset-input.py` (refactor)
**Scope:** M

#### Task 2: Batched inference module
**Description:** Create `src/fpe_inference.py` with the parity-preserving batched inference: load model
via `transform.model_fn`, a `Dataset` that downloads+decodes+eval-transforms each image, and
`run_inference(df, model_dir, storage_bucket, batch_size, num_workers, device) -> df` adding a
`score` column.
**Acceptance criteria:**
- [ ] Reuses `from transform import model_fn`; runs on CPU (`device = torch.device("cpu")`); calls `model.module.forward_single(batch.to(device))` under `torch.no_grad()`.
- [ ] Image decode matches `transform.py` (PIL `Image.open` + `ToTensor`, no convert); per-image failures yield `score = NaN` (not a crash); boto3 S3 client created per DataLoader worker.
- [ ] On a small fixture, scores equal the existing per-image `predict_fn` path to within float tolerance (≤1e-5).
**Verification:** Parity unit test: build a small df, run `run_inference` and the legacy `predict_fn` loop on the same model + images, assert max abs score diff ≤ 1e-5.
**Dependencies:** None (parallel to Task 1)
**Files:** `src/fpe_inference.py` (new), `tests/test_inference_parity.py` (new)
**Scope:** M

#### Checkpoint: Foundation
- [ ] Both modules import cleanly; parity test passes; `imageset-input.py` output unchanged.
- [ ] Review before building the container.

### Phase 2: Container entrypoint

#### Task 3: Self-contained entrypoint `predict-imageset.py`
**Description:** Orchestrator the Batch container runs. Reads job params (CLI args and/or env:
station-id, model-code, imageset-id, batch-size, num-workers) plus DB connection env vars
(`DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD`); looks up the model UUID in the DB and
downloads `model.tar.gz` from the storage bucket (Task 4); fetches `station.json` + `rank-input.json`
from the model bucket into a temp workdir; builds the filtered image df via `fpe_imageset`; extracts
the model; runs `fpe_inference.run_inference`; writes `predictions.csv` (7-col schema) to the imageset
transform S3 key.
**Acceptance criteria:**
- [ ] Runnable as `python src/predict-imageset.py --station-id S --model-code M --imageset-id I` and via env vars (Batch passes either).
- [ ] Validates the imageset belongs to the station; exits non-zero with a clear message on any failure (missing model, no DONE images, empty after filter) so Batch marks the job FAILED.
- [ ] Final `predictions.csv` has columns `split,image_id,timestamp,filename,url,value,score` at the imageset transform key; row count = filtered image count.
**Verification:** End-to-end local run (AWS creds + DB) against a real small imageset; confirm the S3 `predictions.csv` schema and that scores join 1:1 to images. Spot-check scores against the SageMaker path for overlapping images if available.
**Dependencies:** Tasks 1, 2
**Files:** `src/predict-imageset.py` (new)
**Scope:** M

#### Task 4: Resolve the model artifact via the DB `models` table (no `job.txt`, no S3 listing)
**Description:** Look up the model's UUID in the database `models` table by station ID + model code,
then fetch the deployed package from the **storage** bucket at `models/{uuid}/model.tar.gz` (e.g.
`s3://usgs-chs-conte-prod-fpe-storage/models/00abdca8-47df-4afe-a94a-84a7238d480d/model.tar.gz`). The
`models` table has `station_id`, `code`, and `uuid` columns (see [rank-model-db.R](../r/rank-model-db.R#L66-L76)).
The query must return exactly one row.
**Acceptance criteria:**
- [ ] `fetch_model_uuid(conn, station_id, model_code) -> uuid` runs `SELECT uuid FROM models WHERE station_id = %s AND code = %s` and raises a clear error unless **exactly one** row is returned (0 → not found; >1 → ambiguous, operator must disambiguate).
- [ ] `predict-imageset.py` builds the key `s3://usgs-chs-conte-prod-fpe-storage/models/{uuid}/model.tar.gz`, downloads and extracts it for `model_fn`.
**Verification:** For a real station+model code, confirm the query returns one uuid and the storage-bucket `model.tar.gz` downloads and loads; confirm a not-found / duplicate case errors clearly.
**Dependencies:** Task 3 (and Task 1 for the DB connection helpers)
**Files:** `src/fpe_imageset.py` (add `fetch_model_uuid`), `src/predict-imageset.py`
**Scope:** S

#### Checkpoint: Core
- [ ] A local `predict-imageset.py` run produces a correct `predictions.csv` on S3 for a real imageset.

### Phase 3: Packaging

#### Task 5: Dockerfile + container deps
**Description:** Containerize, CPU-only. Slim base + CPU torch wheels; copy only the modules the
inference path needs; install non-torch deps.
**Acceptance criteria:**
- [ ] `Dockerfile` based on `python:3.9-slim`; installs `libgomp1` (OpenMP for torch/sklearn), then CPU torch wheels (`pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu`); copies `src/{transform,modules,datasets,utils,losses,fpe_inference,fpe_imageset,predict-imageset}.py`; `ENTRYPOINT` runs `predict-imageset.py`.
- [ ] `requirements-batch.txt` lists only the non-torch deps (`pandas, numpy<2, Pillow, tqdm, boto3, psycopg2-binary, scikit-learn`) — torch/torchvision are installed separately (CPU index); no sagemaker/pyyaml.
- [ ] `.dockerignore` excludes `.venv/`, `r/`, notebooks, data, `__pycache__`.
- [ ] `docker build` succeeds; in-container `import torch; print(torch.cuda.is_available())` → `False` (CPU build); image markedly smaller than a CUDA base (~1–1.5 GB).
**Verification:** `docker build -t fpe-predict .` then a container run against a tiny imageset producing `predictions.csv` on S3.
**Dependencies:** Task 3
**Files:** `Dockerfile` (new), `.dockerignore` (new), `requirements-batch.txt` (new)
**Scope:** M

#### Task 6: Usage docs + infra contract
**Description:** Document how to build/push the image and the Batch job contract (params/env, and the
IAM + optional resource requirements the job definition must provide). Infra itself is user-managed.
**Acceptance criteria:**
- [ ] README/section covers: build + ECR push commands; the env/args the container expects, including the DB connection env vars (`DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD`); required IAM (S3 read on storage+models, S3 write on models); note the job is CPU-only (no GPU resource needed; Fargate viable).
- [ ] Documents that DB env vars may be injected from Secrets Manager via the job definition `secrets`/`valueFrom` (the `secretsmanager:GetSecretValue` permission then lives on the Batch execution role — infra-side, not in container code).
**Verification:** A colleague can build, push, and submit a job from the doc alone.
**Dependencies:** Task 5
**Files:** `README.md` (or `docs/batch-inference.md`)
**Scope:** S

### Checkpoint: Complete
- [ ] Container scores a real tens-of-thousands-image imageset end-to-end; `predictions.csv` schema/location correct; runtime/cost acceptable.
- [ ] Parity confirmed against the existing model path. Ready for review.

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Batched scores drift from per-image path | High | BN in eval mode (running stats) makes batching exact; Task 2 parity test enforces ≤1e-5 |
| Image decode differs from SageMaker (PIL vs read_image) | High | Mirror `transform.input_fn` exactly: PIL `Image.open` + `ToTensor`, no `.convert` |
| Slim base missing native libs for torch/sklearn (e.g. OpenMP) | Med | Install `libgomp1` via apt; verify `import torch`/`import sklearn` succeed in-container |
| boto3 client shared across forked workers | Med | Create the S3 client lazily per worker (`worker_init_fn` / thread-local) |
| S3 GET throttling on tens of thousands of objects | Med | boto3 adaptive retries; bounded `num_workers`; log/skip-with-NaN on persistent failures |
| DB creds in container | Med | Env vars only — no creds baked into the image, no `r/config.yml`; in Batch, source them from Secrets Manager via the job definition `secrets`/`valueFrom` |
| Long runtime hitting Batch timeout | Low | Tune `--num-workers` (download concurrency) and `--batch-size`; document expected throughput; GPU is the lever if CPU proves too slow |

## Open Questions
- Resource sizing (vCPU/memory) for the Batch job definition — infra-side, not container code.
- How incremental imageset `predictions.csv` files are surfaced to the FPE app vs the single
  per-model `predictions_url` (downstream consumption is out of scope for the container build).

## Verification (end-to-end)
1. **Parity (Task 2):** unit test compares batched `run_inference` vs legacy `predict_fn` on a small
   fixture; max abs diff ≤ 1e-5.
2. **Local entrypoint (Task 3):** run `predict-imageset.py` against a real small imageset with AWS+DB
   creds; confirm S3 `predictions.csv` has the 7-column schema and one score per filtered image.
3. **Container (Task 5):** `docker build`, then run the image against the same imageset; identical
   output to step 2.
4. **Scale:** run on a tens-of-thousands-image imageset; confirm completion, throughput, and that
   per-image failures degrade to `NaN` rather than failing the job.

---

## Reference: key files to reuse
- [src/transform.py](../src/transform.py) — `model_fn` (model load, parity), `input_fn`/`predict_fn` (decode + single-image score reference)
- [src/modules.py](../src/modules.py) — `ResNetRankNet`, `forward_single`
- [src/datasets.py](../src/datasets.py) — image loading reference (note: SageMaker path uses the PIL decode in `transform.py`, not `read_image` here)
- [src/utils.py](../src/utils.py) — `get_url_path`, `load_data`
- [src/imageset-input.py](../src/imageset-input.py) — DB fetch + model-filter logic to extract into `fpe_imageset.py`

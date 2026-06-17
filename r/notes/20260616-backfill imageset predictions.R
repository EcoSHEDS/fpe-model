# loop through models to backfill imageset predictions
#   for each model:
#     fetch predictions.csv from s3 model bucket
#     get imageset + images manifest from database
#     group predictions by imageset
#     write imageset predictions to s3 model bucket

library(tidyverse)
library(glue)
library(paws)
library(furrr)
library(janitor)

S3_MODEL_BUCKET <- "usgs-chs-conte-prod-fpe-models"

cache_dir <- "notes/20260616-backfill imageset predictions"
stopifnot(dir.exists(cache_dir))


# connect -----------------------------------------------------------------

config_db <- config::get("db")

con <- DBI::dbConnect(
  RPostgres::Postgres(),
  host = config_db$host,
  port = config_db$port,
  dbname = "postgres",
  user = config_db$user,
  password = config_db$password
)

Sys.setenv(AWS_PROFILE = "conte-prod")
s3 <- paws::s3()
s3$list_buckets()


# fetch: db ---------------------------------------------------------------

db_models <- tbl(con, "models") |>
  collect()
db_imagesets <- tbl(con, "imagesets") |>
  filter(station_id %in% db_models$station_id) |>
  collect()

db <- list(
  models = db_models,
  imagesets = db_imagesets
)

write_rds(db, file.path(cache_dir, "db.rds"))
db <- read_rds(file.path(cache_dir, "db.rds"))


# fetch: model predictions -------------------------------------------------

read_s3_csv <- function (bucket, key) {
  obj <- s3$get_object(
    Bucket = bucket,
    Key = key
  )

  raw <- rawToChar(obj$Body)
  read_csv(I(raw), col_types = cols(.default = col_guess(), timestamp = col_character()))
}
read_s3_csv <- possibly(read_s3_csv, otherwise = NULL)

plan(multisession, workers = 6)
models <- db$models |>
  mutate(
    predictions_key = glue::glue("rank/{station_id}/models/{code}/transform/predictions.csv"),
    predictions = future_map(predictions_key, \(x) read_s3_csv(bucket = S3_MODEL_BUCKET, key = x), .progress = TRUE)
  ) |>
  filter(status == "DONE")
write_rds(models, file.path(cache_dir, "models.rds"))
models <- read_rds(file.path(cache_dir, "models.rds"))


# generate: imageset predictions ------------------------------------------

imagesets <- models |>
  select(
    model_id = id,
    station_id,
    model_code = code,
    predictions
  ) |>
  unnest(predictions) |>
  mutate(
    imageset_uuid = str_extract(filename, "(?<=imagesets/)[^/]+")
  ) |>
  select(-value) |>
  nest(.by = c(model_id, station_id, model_code, imageset_uuid), .key = "predictions") |>
  mutate(
    # rank/29/models/RANK-FLOW-20240410/transform/imagesets/5a53b364-7a42-4708-b66e-d837c6b05f3e/predictions.csv
    predictions_key = glue::glue("rank/{station_id}/models/{model_code}/transform/imagesets/{imageset_uuid}/predictions.csv")
  )
write_rds(imagesets, file.path(cache_dir, "imagesets.rds"))
imagesets <- read_rds(file.path(cache_dir, "imagesets.rds"))

# predictions do not include PII
imagesets |>
  mutate(
    n_predictions = map_int(predictions, nrow)
  ) |>
  left_join(
    db$imagesets |>
      select(imageset_uuid = uuid, n_images),
    by = "imageset_uuid"
  ) |>
  filter(n_predictions != n_images) |>
  tabyl(station_id)


# export: imageset predictions to S3 --------------------------------------

library(paws)
library(readr)

# Helper to produce consistent CSV output
format_csv <- function(df) {
  tmp <- tempfile(fileext = ".csv")
  write_csv(df, tmp)
  paste(readLines(tmp), collapse = "\n")
}

write_s3_csv <- function(data, bucket, key) {
  # Convert data frame to CSV text
  csv_text <- format_csv(data)

  # Convert to raw vector for S3 upload
  raw_body <- charToRaw(csv_text)

  # Upload to S3
  s3$put_object(
    Bucket = bucket,
    Key = key,
    Body = raw_body,
    ContentType = "text/csv"
  )

  invisible(TRUE)
}

plan(multisession, workers = 6)
imagesets_saved <- imagesets |>
  mutate(
    success = future_map2_lgl(predictions, predictions_key, function (data, key) {
      write_s3_csv(
        data = data,
        bucket = S3_MODEL_BUCKET,
        key = key
      )
    }, .progress = TRUE)
  )

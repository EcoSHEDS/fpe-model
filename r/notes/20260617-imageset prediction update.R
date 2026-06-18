# update predictions.csv for sites with new imageset predictions

library(tidyverse)
library(glue)
library(paws)
library(furrr)
library(janitor)

S3_STORAGE_BUCKET <- "usgs-chs-conte-prod-fpe-storage"
S3_MODEL_BUCKET <- "usgs-chs-conte-prod-fpe-models"

cache_dir <- "notes/20260617-imageset prediction update"
stopifnot(dir.exists(cache_dir))

station_ids_flow <- c(
  75,
  273,
  73,
  272,
  271,
  72,
  74,
  53,
  49,
  50,
  # 274, no model exists, insufficient annotations
  51,
  52,
  47,
  48,
  46,
  54,
  44,
  45
)
station_ids_snow <- c(280, 303)
stations_variables <- bind_rows(
  FLOW_CFS = tibble(station_id = station_ids_flow),
  SNOW_FT = tibble(station_id = station_ids_snow),
  .id = "variable_id"
)
stopifnot(!anyDuplicated(stations_variables$station_id))

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
db_stations <- tbl(con, "stations") |>
  filter(id %in% db_models$station_id) |>
  collect()

# merge: models + imagesets -----------------------------------------------

models <- stations_variables |>
  left_join(
    db_models,
    by = c("station_id", "variable_id")
  )

stopifnot(nrow(stations_variables) == nrow(models))

models_imagesets <- models |>
  left_join(
    db_imagesets |>
      select(station_id, imageset_uuid = uuid, imageset_status = status, n_images),
    by = "station_id"
  )

summary(models_imagesets$n_images)


# fetch: imageset predictions ---------------------------------------------

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
models_imagesets_predictions <- models_imagesets |>
  mutate(
    predictions_key = glue::glue("rank/{station_id}/models/{code}/transform/imagesets/{imageset_uuid}/predictions.csv"),
    predictions = future_map(predictions_key, \(x) read_s3_csv(bucket = S3_MODEL_BUCKET, key = x), .progress = TRUE),
    predictions_exist = map_lgl(predictions, \(x) !is.null(x))
  )

models_imagesets_predictions |>
  group_by(station_id, model_code = code) |>
  summarise(
    n_imagesets = n(),
    n_predictions = sum(predictions_exist),
    n_missing = sum(!predictions_exist)
  )

# # imagesets with missing predictions
# models_imagesets_predictions |>
#   filter(!predictions_exist) |>
#   select(-predictions) |>
#   view()
#
#
# models_imagesets_predictions |>
#   filter(!predictions_exist) |>
#   select(station_id, model_code = code, imageset_uuid) |>
#   group_by(station_id, model_code) |>
#   summarise(
#     imageset_uuid = str_c(imageset_uuid, collapse = ",")
#   ) |>
#   write_delim(file.path(cache_dir, "model-imagesets.txt"),col_names = FALSE)


# fetch: model predictions ------------------------------------------------

models_pred_train <- models |>
  mutate(
    predictions_key = glue::glue("rank/{station_id}/models/{code}/transform/predictions.csv"),
    predictions = future_map(predictions_key, \(x) read_s3_csv(bucket = S3_MODEL_BUCKET, key = x), .progress = TRUE)
  )


# generate: model predictions from imageset predictions -------------------

models_pred_imagesets <- models_imagesets_predictions |>
  select(
    model_id = id, model_code = code, model_uuid = uuid, station_id,
    predictions
  ) |>
  unnest(predictions) |>
  select(-split, -value) |>
  arrange(model_id, timestamp) |>
  nest(.by = c(model_id, model_code, model_uuid, station_id), .key = "predictions")


# compare: train vs imageset predictions ----------------------------------

models_pred_compare <- bind_rows(
  train = models_pred_train |>
    select(
      model_id = id, model_code = code, model_uuid = uuid, station_id,
      predictions
    ) |>
    unnest(predictions) |>
    select(-split, -value),
  imageset = models_pred_imagesets |>
    unnest(predictions),
  .id = "source"
)

models_pred_compare |>
  pivot_wider(names_from = "source", values_from = "score") |>
  ggplot(aes(train, imageset)) +
  geom_point(size = 0.2)

stopifnot(
  models_pred_compare |>
    pivot_wider(names_from = "source", values_from = "score") |>
    mutate(diff = abs(train - imageset)) |>
    filter(diff > 0.1) |>
    nrow() == 0
)


# export: model predictions.csv -------------------------------------------

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

models_pred_saved <- models_pred_imagesets |>
  mutate(
    success = future_map2_lgl(predictions, model_uuid, function (data, uuid) {
      predictions_key <- glue::glue("models/{uuid}/predictions.csv")
      # s3$copy_object(
      #   Bucket = S3_STORAGE_BUCKET,
      #   CopySource = paste0(S3_STORAGE_BUCKET, "/", predictions_key),
      #   Key = glue::glue("models/{uuid}/predictions-train.csv")
      # )
      write_s3_csv(
        data = data,
        bucket = S3_STORAGE_BUCKET,
        key = predictions_key
      )
    }, .progress = TRUE)
  )


# check that these all have the old train predictions
models_pred_saved_check <- models_pred_saved |>
  mutate(
    predictions_old_key = glue::glue("models/{model_uuid}/predictions-train.csv"),
    predictions_old = future_map(predictions_old_key, \(x) read_s3_csv(bucket = S3_STORAGE_BUCKET, key = x), .progress = TRUE),
  )

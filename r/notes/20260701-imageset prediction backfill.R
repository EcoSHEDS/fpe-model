# create list of stations + models + imagesets for imageset prediction jobs

library(tidyverse)
library(glue)
library(paws)
library(furrr)
library(janitor)

S3_MODEL_BUCKET <- "usgs-chs-conte-prod-fpe-models"

cache_dir <- "notes/20260701-imageset prediction backfill"
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


# load: db ----------------------------------------------------------------

db_models <- tbl(con, "models") |>
  filter(default) |>
  collect()
db_imagesets <- tbl(con, "imagesets") |>
  filter(station_id %in% db_models$station_id) |>
  collect()
db_datasets <- tbl(con, "datasets") |>
  filter(station_id %in% db_models$station_id) |>
  collect()
db_series <- tbl(con, "series") |>
  filter(dataset_id %in% db_datasets$id) |>
  collect()
db_stations <- tbl(con, "stations") |>
  filter(id %in% db_models$station_id) |>
  collect()



# extract: stations with models --------------------------------------------
# list stations and models with obs. data

datasets <- db_datasets |>
  filter(status == "DONE") |>
  transmute(
    dataset_id = id,
    station_id,
    dataset_uuid = uuid,
    dataset_start_timestamp = start_timestamp,
    dataset_end_timestamp = end_timestamp,
    dataset_n_rows = n_rows
  ) |>
  left_join(
    db_series |>
      select(dataset_id, variable_id),
    by = c("dataset_id")
  )

stations <- db_stations |>
  transmute(
    station_id = id,
    user_id,
    station_name = name,
    latitude,
    longitude,
    timezone,
    private,
    nwis_id,
    waterbody_type = as.character(waterbody_type),
    station_status = as.character(status)
  )

stations_obs_nwis <- stations |>
  filter(!is.na(na_if(nwis_id, ""))) |>
  mutate(
    variable_id = "FLOW_CFS"
  )

stations_obs_fpe <- stations |>
  filter(!station_id %in% stations_obs_nwis$station_id) |>
  inner_join(
    datasets |>
      group_by(station_id, variable_id) |>
      summarise(
        n_datasets = n(),
        dataset_start_timestamp = min(dataset_start_timestamp),
        dataset_end_timestamp = max(dataset_end_timestamp),
        dataset_n_rows = sum(dataset_n_rows),
        .groups = "drop"
      ),
    by = "station_id"
  )

stations_obs <- bind_rows(
  nwis = stations_obs_nwis,
  fpe = stations_obs_fpe,
  .id = "obs_source"
)

stopifnot(
  stations_obs |>
    add_count(station_id, variable_id) |>
    filter(n > 1) |>
    nrow() == 0,
  all(!is.na(stations_obs$obs_source))
)

models <- db_models |>
  filter(status == "DONE") |>
  left_join(
    stations_obs |>
      select(station_id, variable_id, obs_source),
    by = c("station_id", "variable_id")
  ) |>
  transmute(
    model_id = id,
    station_id,
    variable_id,
    obs_source,
    model_code = code,
    model_uuid = uuid,
    model_prefix = str_c("rank", station_id, "models", model_code, sep = "/")
  ) |>
  print()

models_imagesets <- models |>
  left_join(
    db_imagesets |>
      filter(status == "DONE") |>
      select(
        station_id,
        imageset_id = id,
        imageset_uuid = uuid,
        n_images,
        start_timestamp, end_timestamp
      ),
    by = "station_id",
    relationship = "many-to-many"
  ) |>
  mutate(
    imageset_prefix = str_c(model_prefix, "transform", "imagesets", imageset_uuid, sep = "/")
  ) |>
  print()

# result <- s3$list_objects(Bucket = S3_MODEL_BUCKET, Prefix = paste0(models_imagesets$imageset_prefix[[1]], 'a'), MaxKeys=1)
# length(result$Contents)
# result$Contents[[1]]

check_imageset_predictions_csv <- function (imageset_prefix) {
  key <- str_c(imageset_prefix, "predictions.csv", sep = "/")
  cat(key, "\n")
  s3$head_object(
    Bucket = S3_MODEL_BUCKET,
    Key = key
  )
}
possibly_check_imageset_predictions_csv <- possibly(check_imageset_predictions_csv)

# possibly_check_imageset_predictions_csv(models_imagesets$imageset_prefix[[1]])

plan(multisession, workers = 6)
models_imageset_predictions_csv <- models_imagesets |>
  mutate(
    predictions_csv = future_map(imageset_prefix, \(x) possibly_check_imageset_predictions_csv(x), .progress = TRUE),
    predictions_csv_content_length = map_dbl(predictions_csv, function(x) {
      if (is.null(x)) return(0)
      x$ContentLength
    })
  )

models_imageset_predictions_csv_exists <- models_imageset_predictions_csv |>
  filter(predictions_csv_content_length > 0)
models_imageset_predictions_csv_missing <- models_imageset_predictions_csv |>
  filter(predictions_csv_content_length == 0)

models_imageset_predictions_csv_missing
# n=1280

models_imageset_predictions_csv_missing |>
  tabyl(obs_source)


# export: manifest --------------------------------------------------------

manifest <- models_imageset_predictions_csv_missing |>
  summarise(imageset_uuid = str_c(imageset_uuid, collapse = ","), .by = c(station_id, model_code))


manifest |>
  write_delim(file.path(cache_dir, "model-imagesets.txt"), col_names = FALSE)

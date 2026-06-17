# create list of stations + models + imagesets for imageset prediction jobs

library(tidyverse)
library(glue)
library(paws)
library(furrr)
library(janitor)

S3_MODEL_BUCKET <- "usgs-chs-conte-prod-fpe-models"

cache_dir <- "notes/20260616-imageset prediction jobs"
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
      select(station_id, imageset_uuid = uuid),
    by = "station_id"
  )


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

models_imagesets_predictions_missing <- models_imagesets_predictions |>
  filter(!predictions_exist)

models_imagesets_predictions_missing |>
  tabyl(station_id, variable_id)

models_imagesets_predictions_missing |>
  select(station_id, model_code = code, imageset_uuid) |>
  group_by(station_id, model_code) |>
  summarise(
    imageset_uuid = str_c(imageset_uuid, collapse = ",")
  ) |>
  write_delim(file.path(cache_dir, "model-imagesets.txt"),col_names = FALSE)


# confirm -----------------------------------------------------------------

plan(multisession, workers = 6)
station_45_imagesets <- models_imagesets |>
  filter(station_id == 45) |>
  mutate(
    predictions_key = glue::glue("rank/{station_id}/models/{code}/transform/imagesets/{imageset_uuid}/predictions.csv"),
    predictions = future_map(predictions_key, \(x) read_s3_csv(bucket = S3_MODEL_BUCKET, key = x), .progress = TRUE),
    predictions_exist = map_lgl(predictions, \(x) !is.null(x))
  )

station_45_imagesets |>
  tabyl(predictions_exist)

station_45_pred_iset <- station_45_imagesets |>
  unnest(predictions) |>
  left_join(
    db_stations |>
      select(station_id = id, timezone),
    by = "station_id"
  ) |>
  mutate(
    timestamp = ymd_hms(timestamp),
    timestamp = with_tz(timestamp, tzone = timezone)
  )

station_45_pred_iset |>
  ggplot(aes(timestamp, score)) +
  geom_line()
station_45_predictions$uuid |> unique()


names(station_45_imagesets$predictions[[1]])

station_45_pred_train <- read_s3_csv(
  bucket = S3_MODEL_BUCKET,
  key = glue::glue("rank/45/models/{station_45_imagesets$code[[1]]}/transform/predictions.csv")
) |>
  mutate(
    timestamp = ymd_hms(timestamp),
    timestamp = with_tz(timestamp, tzone = "America/New_York")
  )

station_45_pred_comp <- bind_rows(
  iset = station_45_pred_iset |>
    select(split, image_id, timestamp, filename, score),
  train = station_45_pred_train |>
    select(split, image_id, timestamp, filename, score),
  .id = "source"
)
station_45_pred_comp |>
  tabyl(split, source)

station_45_pred_comp |>
  select(-split) |>
  pivot_wider(names_from = "source", values_from = "score") |>
  # filter(iset != train)
  ggplot(aes(train, iset)) +
  geom_point()

station_45_pred_comp |>
  ggplot(aes(timestamp, score, color = source)) +
  geom_line()

station_45_pred_iset |>
  arrange(timestamp) |>
  mutate(
    timestamp = with_tz(timestamp, "UTC"),
    timestamp = format_ISO8601(timestamp, usetz = TRUE)
  ) |>
  select(image_id, timestamp, filename, url, score) |>
  write_csv("predictions.csv")

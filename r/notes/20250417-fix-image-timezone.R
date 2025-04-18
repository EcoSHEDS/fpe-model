# fix utc offset for images at stations 76, 77, 78, 79
# shift all images in imagesets with utcOffset=-4 forward 1 hour (should have been UTC-5=EST)
# fix predictions.csv file too so results appear correct on FPE

library(tidyverse)
library(bit64)

station_ids <- c(76, 77, 78, 79)

config <- config::get()

con <- DBI::dbConnect(
  RPostgres::Postgres(),
  host = config$db$host,
  port = config$db$port,
  dbname = config$db$database,
  user = config$db$user,
  password = config$db$password
)


# get images -----------------------------------------------------------

imagesets <- DBI::dbGetQuery(
  con,
  "select id, station_id, config->'timestamp'->'utcOffset' as utcOffset from imagesets"
) %>%
  filter(station_id %in% station_ids)

# note: timestamp already fixed
images <- tbl(con, "images") %>%
  filter(imageset_id %in% imagesets$id) %>%
  select(id, imageset_id, filename, timestamp) %>%
  collect()

nrow(images)



# update predictions files ------------------------------------------------

model_code <- "RANK-FLOW-20250417"
root_dir <- "/mnt/d/fpe/rank/20250417"

station_id <- 77

for (station_id in c(77, 78, 79)) {
  station_dir <- file.path(root_dir, station_id)
  stopifnot(dir.exists(station_dir))

  images_file <- file.path(station_dir, "models", model_code, "input", "images.csv")
  images <- read_csv(images_file)

  predictions_file <- file.path(station_dir, "models", model_code, "transform", "predictions.csv")
  predictions <- read_csv(predictions_file)

  predictions %>%
    write_csv(paste0(predictions_file, ".bak"))

  predictions_updated <- predictions %>%
    left_join(
      select(images, image_id, new_timestamp = timestamp, new_value = value),
      by = "image_id"
    )

  # predictions_updated %>%
  #   filter(value != new_value)
  predictions_updated %>%
    mutate(timestamp = new_timestamp, value = new_value) %>%
    select(-new_timestamp, -new_value) %>%
    write_csv(predictions_file)
}

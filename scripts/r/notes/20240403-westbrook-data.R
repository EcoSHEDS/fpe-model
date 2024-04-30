# set up model inputs for Westbrook stations

Sys.setenv(TZ = "US/Eastern")

library(tidyverse)
library(lubridate)
library(jsonlite)
library(glue)

FPE_DIR <- "/mnt/d/fpe/rank"
DATASET_VERSION <- "RANK-FLOW-20240402"

x <- tibble(
  dir = list.dirs(FPE_DIR, recursive = FALSE)
) %>%
  mutate(
    station_json = map(dir, function (x) {
      read_json(file.path(x, "datasets", DATASET_VERSION, "station.json"))
    }),
    station_id = map_int(station_json, \(x) x$id),
    station_name = map_chr(station_json, \(x) x$name),
    nwis_id = map_chr(station_json, function (x) {
      if (is.null(x$nwis_id) || x$nwis_id == "") return(NA_character_)
      x$nwis_id
    }),
    timezone = map_chr(station_json, \(x) x$timezone),
    annotations = map(dir, function (x) {
      read_csv(file.path(x, "datasets", DATASET_VERSION, "annotations.csv"), show_col_types = FALSE)
    }),
    images = map(dir, function (x) {
      read_csv(file.path(x, "datasets", DATASET_VERSION, "images.csv"), show_col_types = FALSE)
    }),
    values = map(dir, function (x) {
      f <- file.path(x, "datasets", DATASET_VERSION, "values.csv")
      if (!file.exists(f)) return(tibble())
      read_csv(f, show_col_types = FALSE)
    })
  )


x %>%
  mutate(
    values_start = map_dbl(values, function (values) {
      if (nrow(values) == 0) return(NA_real_)
      values %>%
        filter(!is.na(value)) %>%
        pull(timestamp) %>%
        min() %>%
        as.numeric()
    }),
    values_end = map_dbl(values, function (values) {
      if (nrow(values) == 0) return(NA_real_)
      values %>%
        filter(!is.na(value)) %>%
        pull(timestamp) %>%
        max() %>%
        as.numeric()
    }),
    values_start = as_date(with_tz(as.POSIXct(values_start), tzone = timezone)),
    values_end = as_date(with_tz(as.POSIXct(values_end), tzone = timezone))
  ) %>%
  arrange(desc(values_end)) %>%
  select(station_id, station_name, nwis_id, values_start, values_end)


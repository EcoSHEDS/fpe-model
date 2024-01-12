# generate model input from annotations
# usage: Rscript model-input.R <path to station folder> <model id>
# example: Rscript model-input.R D:/fpe/sites/AVERYBB 20230928

Sys.setenv(TZ = "GMT")

library(tidyverse)
library(lubridate)
library(jsonlite)
library(logger)


args <- commandArgs(trailingOnly = TRUE)
site_dir <- path.expand(args[1])
model_id <- args[2]

input_dir <- file.path(site_dir, "models", model_id, "input")
dir.create(input_dir, showWarnings = FALSE, recursive = TRUE)

log_info("site_dir: {site_dir}")
log_info("model_id: {model_id}")
log_info("input_dir: {input_dir}")

# functions ---------------------------------------------------------------

split_pairs <- function (x, seed = 1691) {
  x2 <- x %>%
    rowwise() %>%
    mutate(
      min_timestamp = min(left.timestamp, right.timestamp),
      max_timestamp = max(left.timestamp, right.timestamp),
      start_year =  year(as_date(min_timestamp)),
      end_year =  year(as_date(max_timestamp))
    )
  test_cutoff <- quantile(x2$min_timestamp, probs = 0.9)
  pairs <- x2 %>%
    ungroup() %>%
    transmute(
      image_id_1 = left.imageId,
      timestamp_1 = left.timestamp,
      filename_1 = left.filename,
      label_1 = left.flow_cfs,
      image_id_2 = right.imageId,
      timestamp_2 = right.timestamp,
      filename_2 = right.filename,
      label_2 = right.flow_cfs,
      pair_label = case_when(
        rank == "SAME" ~ 0,
        rank == "LEFT" ~ 1,
        rank == "RIGHT" ~ -1
      ),
      pair = row_number()
    )

  pairs_test <- pairs %>%
    filter(timestamp_1 >= test_cutoff, timestamp_2 >= test_cutoff)

  set.seed(seed)
  pairs_train_val <- pairs %>%
    filter(timestamp_1 < test_cutoff, timestamp_2 < test_cutoff) %>%
    mutate(
      split = if_else(runif(n()) < 0.2, "val", "train")
    )
  pairs_train <- pairs_train_val %>%
    filter(split == "train") %>%
    select(-split)
  pairs_val <- pairs_train_val %>%
    filter(split == "val") %>%
    select(-split)

  list(
    train = duplicate_pairs(pairs_train),
    val = duplicate_pairs(pairs_val),
    test = duplicate_pairs(pairs_test)
  )
}

duplicate_pairs <- function (x) {
  x2 <- bind_cols(
    select(x, pair),
    select(x, ends_with("_2")) %>%
      rename_with(~ str_replace(., "2", "1")),
    select(x, ends_with("_1")) %>%
      rename_with(~ str_replace(., "1", "2")),
    transmute(x, pair_label = -1 * pair_label)
  )
  bind_rows(x, x2) %>%
    arrange(pair)
}

export_input <- function(images, annotations, pairs, dir, notes) {
  images %>%
    write_csv(file.path(dir, "images.csv"))

  annotations %>%
    write_csv(file.path(dir, "annotations.csv"))

  pairs$train %>%
    write_csv(file.path(dir, "pairs-train.csv"))
  pairs$val %>%
    write_csv(file.path(dir, "pairs-val.csv"))
  pairs$test %>%
    write_csv(file.path(dir, "pairs-test.csv"))

  manifest <- bind_rows(
    pairs$train,
    pairs$val,
    pairs$test
  ) %>%
    select(filename_1, filename_2) %>%
    pivot_longer(everything()) %>%
    pull(value) %>%
    unique()

  manifest_json <- str_replace(
    toJSON(manifest, auto_unbox = TRUE),
    "\\[", "[{\"prefix\": \"s3://usgs-chs-conte-prod-fpe-storage/\"},"
  )
  write_file(manifest_json, file.path(input_dir, "manifest.json"))

  list(
    method = "human",
    num_train_pairs = nrow(pairs$train) / 2,
    num_val_pairs = nrow(pairs$val) / 2,
    num_test_pairs = nrow(pairs$test) / 2,
    notes = notes
  ) %>%
    write_json(file.path(dir, "args.json"), auto_unbox = TRUE)
}

# run ---------------------------------------------------------------------

log_info("loading station.json")
station <- read_json(file.path(site_dir, "data", "station.json"))

log_info("loading images.csv")
images <- read_csv(file.path(site_dir, "data", "images.csv"), show_col_types = FALSE) %>%
  mutate(
    timestamp = with_tz(timestamp, tzone = station$timezone)
  ) %>%
  filter(
    hour(timestamp) %in% 7:18 # daytime only
  )

log_info("loading annotations.csv")
annotations <- read_csv(file.path(site_dir, "data", "annotations.csv"), show_col_types = FALSE) %>%
  filter(
    rank != "UNKNOWN",
    left.imageId %in% images$image_id,
    right.imageId %in% images$image_id
    #user_id != "9bc0d5e3-a871-4ffa-ae4f-1cbc44b0ea17"
  )

log_info("splitting pairs")
pairs <- split_pairs(annotations)

log_info("exporting: {input_dir}")
#export_input(images, annotations, pairs, dir = input_dir, notes = "exclude user_id=9bc0d5e3, exclude nighttime (7PM-7AM)")
export_input(images, annotations, pairs, dir = input_dir, notes = "exclude nighttime (7PM-7AM)")

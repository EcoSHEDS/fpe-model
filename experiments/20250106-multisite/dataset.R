# generate dataset for 20250106-multisite

library(tidyverse)
library(jsonlite)
library(janitor)
library(glue)

# stations:
#   10 WB LOWER
#   16 WB RESERVOIR
#   29 WB ZERO

root_dir <- "/mnt/d/fpe/experiments/"
code <- "20250106-multisite"
exp_dir <- file.path(root_dir, code)
station_ids <- read_lines(file.path(exp_dir, "stations", "stations.txt"))

images <- tibble(
  station_id = station_ids
) %>%
  rowwise() %>%
  mutate(
    # station = list(read_json(file.path(exp_dir, "stations", station_id, "datasets", dataset_code, "station.json"))),
    images = list({
      read_csv(file.path(exp_dir, "stations", station_id, "datasets", dataset_code, "images.csv")) %>%
        filter(!is.na(value)) %>%
        mutate(value = log10(value))
    }),
    start = min(images$timestamp),
    end = max(images$timestamp),
    n_images = nrow(images)
  )

images %>%
  select(images) %>%
  unnest(images) %>%
  ggplot(aes(timestamp, value)) +
  geom_point(aes(color = station_name), size = 0.5)


# run: 01 -----------------------------------------------------------------
# small dataset (n=1000 per site, 3 sites)

images_01 <- images %>%
  mutate(
    images = list({
      slice_sample(images, n = 1000)
    })
  ) %>%
  select(images) %>%
  unnest(images) %>%
  mutate(
    r = runif(n = n()),
    split = case_when(
      station_id == 29 ~ "test-out",
      timestamp >= ymd(20230101) ~ "test-in",
      r < 0.8 ~ "train",
      TRUE ~ "val"
    )
  )

images_01 %>%
  ggplot(aes(timestamp, value)) +
  geom_point(aes(color = split), size = 0.5)

images_01 %>%
  write_csv(file.path(exp_dir, "runs", "01", "input", "images.csv"))

images_01 %>%
  pull(filename) %>%
  unique() %>%
  toJSON(auto_unbox = TRUE) %>%
  str_replace(
    "\\[", "[{\"prefix\": \"s3://usgs-chs-conte-prod-fpe-storage/\"},"
  ) %>%
  write_file(file.path(exp_dir, "runs", "01", "input", "manifest.json"))

metrics_01 <- read_csv(file.path(exp_dir, "runs", "01", "output", "metrics.csv"))

metrics_01 %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, color = "train")) +
  geom_line(aes(y = val_loss, color = "val"))

pred_01 <- read_csv(file.path(exp_dir, "runs", "01", "output", "predictions.csv"))

pred_01 %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)

pred_01 %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = split), size = 0.5) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)


# run: 02 -----------------------------------------------------------------
# larger dataset (n=10000, 3 sites)

images_02 <- images %>%
  mutate(
    images = list({
      slice_sample(images, n = 10000, replace = FALSE)
    })
  ) %>%
  select(images) %>%
  unnest(images) %>%
  mutate(
    r = runif(n = n()),
    split = case_when(
      station_id == 29 ~ "test-out",
      timestamp >= ymd(20230101) ~ "test-in",
      r < 0.8 ~ "train",
      TRUE ~ "val"
    )
  )

images_02 %>%
  ggplot(aes(timestamp, value)) +
  geom_point(aes(color = split), size = 0.5)

dir.create(file.path(exp_dir, "runs", "02", "input"), showWarnings = FALSE, recursive = TRUE)
images_02 %>%
  write_csv(file.path(exp_dir, "runs", "02", "input", "images.csv"))

images_02 %>%
  pull(filename) %>%
  unique() %>%
  toJSON(auto_unbox = TRUE) %>%
  str_replace(
    "\\[", "[{\"prefix\": \"s3://usgs-chs-conte-prod-fpe-storage/\"},"
  ) %>%
  write_file(file.path(exp_dir, "runs", "02", "input", "manifest.json"))

metrics_02 <- read_csv(file.path(exp_dir, "runs", "02", "output", "metrics.csv"))

metrics_02 %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, color = "train")) +
  geom_line(aes(y = val_loss, color = "val"))

pred_02 <- read_csv(file.path(exp_dir, "runs", "02", "output", "predictions.csv"))

pred_02 %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)

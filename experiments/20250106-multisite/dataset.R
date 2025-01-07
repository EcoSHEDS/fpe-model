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
      read_csv(file.path(exp_dir, "stations", station_id, "datasets", code, "images.csv")) %>%
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


# run 03 ------------------------------------------------------------------
# finetune WB0 using run 02 model

images_03 <- read_csv(file.path(exp_dir, "runs", "02", "input", "images.csv")) %>%
  filter(station_id == 29)
images_03 %>%
  write_csv(file.path(exp_dir, "runs", "03", "input", "images.csv"))

pairs_03_all <- read_csv(file.path(exp_dir, "stations", "29", "pairs.csv")) %>%
  mutate(
    across(starts_with("value"), log10)
  ) %>%
  filter(!is.na(value_1), !is.na(value_2))
n_train <- 500
n_val <- n_train / 0.8 - n_train
set.seed(1118)

pairs_03_train <- pairs_03_all %>%
  filter(split == "train") %>%
  nest_by(split, pair) %>%
  ungroup() %>%
  slice_sample(n = n_train, replace = FALSE) %>%
  unnest(data)
pairs_03_val <- pairs_03_all %>%
  filter(split == "val") %>%
  nest_by(split, pair) %>%
  ungroup() %>%
  slice_sample(n = n_val, replace = FALSE) %>%
  unnest(data)

pairs_03 <- bind_rows(pairs_03_train, pairs_03_val)

summary(pairs_03)

pairs_03 %>%
  write_csv(file.path(exp_dir, "runs", "03", "input", "pairs.csv"))


metrics_03 <- read_csv(file.path(exp_dir, "runs", "03", "output", "data", "metrics.csv"))

metrics_03 %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, color = "train")) +
  geom_line(aes(y = val_loss, color = "val"))

pred_03 <- read_csv(file.path(exp_dir, "runs", "03", "output", "data", "predictions.csv"))

pred_03 %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)


# run 04 ------------------------------------------------------------------
# untrained resnet

dir.create(file.path(exp_dir, "runs", "04", "input"), recursive = TRUE, showWarnings = FALSE)
file.copy(
  file.path(exp_dir, "runs", "03", "input", "images.csv"),
  file.path(exp_dir, "runs", "04", "input", "images.csv"),
  overwrite = TRUE
)

pred_04 <- read_csv(file.path(exp_dir, "runs", "04", "output", "data", "predictions.csv"))

pred_04 %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)


# run 05 ------------------------------------------------------------------
# train resnet from scratch using same files as run 03

dir.create(file.path(exp_dir, "runs", "05", "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(exp_dir, "runs", "05", "output", "model"), recursive = TRUE, showWarnings = FALSE)
file.copy(
  file.path(exp_dir, "runs", "03", "input", "pairs.csv"),
  file.path(exp_dir, "runs", "05", "input", "pairs.csv"),
  overwrite = TRUE
)
file.copy(
  file.path(exp_dir, "runs", "03", "input", "images.csv"),
  file.path(exp_dir, "runs", "05", "input", "images.csv")
)

pred_05 <- read_csv(file.path(exp_dir, "runs", "05", "output", "data", "predictions.csv"))

pred_05 %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)

bind_rows(
  `rank` = pred_05,
  `reg+rank` = pred_03,
  .id = "model"
) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = model), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)

bind_rows(
  `rank(500)` = pred_05,
  `reg+rank(500)` = pred_03,
  .id = "model"
) %>%
  group_by(model) %>%
  mutate(
    across(c(value, prediction), rank)
  ) %>%
  ungroup() %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = model), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)

bind_rows(
  `rank(500)` = pred_05,
  `reg+rank(500)` = pred_03,
  .id = "model"
) %>%
  group_by(model) %>%
  summarize(
    tau = cor(value, prediction, method = "kendall")
  )

# run 06 ------------------------------------------------------------------
# same as run 03 but with 2000 pairs

dir.create(file.path(exp_dir, "runs", "06", "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(exp_dir, "runs", "06", "output", "model"), recursive = TRUE, showWarnings = FALSE)
file.copy(
  file.path(exp_dir, "runs", "03", "input", "model.pth"),
  file.path(exp_dir, "runs", "06", "input", "model.pth")
)
file.copy(
  file.path(exp_dir, "runs", "03", "input", "images.csv"),
  file.path(exp_dir, "runs", "06", "input", "images.csv")
)
n_train <- 2000
n_val <- n_train / 0.8 - n_train
set.seed(1118)

pairs_06_train <- pairs_03_all %>%
  filter(split == "train") %>%
  nest_by(split, pair) %>%
  ungroup() %>%
  slice_sample(n = n_train, replace = FALSE) %>%
  unnest(data)
pairs_06_val <- pairs_03_all %>%
  filter(split == "val") %>%
  nest_by(split, pair) %>%
  ungroup() %>%
  slice_sample(n = n_val, replace = FALSE) %>%
  unnest(data)

pairs_06 <- bind_rows(pairs_06_train, pairs_06_val)

summary(pairs_06)

pairs_06 %>%
  write_csv(file.path(exp_dir, "runs", "06", "input", "pairs.csv"))

metrics_06 <- read_csv(file.path(exp_dir, "runs", "06", "output", "data", "metrics.csv"))

metrics_06 %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, color = "train")) +
  geom_line(aes(y = val_loss, color = "val"))

pred_06 <- read_csv(file.path(exp_dir, "runs", "06", "output", "data", "predictions.csv"))

pred_06 %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)


# run 07 ------------------------------------------------------------------
# train ranknet using run 06 pairs

dir.create(file.path(exp_dir, "runs", "07", "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(exp_dir, "runs", "07", "output", "model"), recursive = TRUE, showWarnings = FALSE)
file.copy(
  file.path(exp_dir, "runs", "06", "input", "pairs.csv"),
  file.path(exp_dir, "runs", "07", "input", "pairs.csv")
)
file.copy(
  file.path(exp_dir, "runs", "06", "input", "images.csv"),
  file.path(exp_dir, "runs", "07", "input", "images.csv")
)

metrics_07 <- read_csv(file.path(exp_dir, "runs", "07", "output", "data", "metrics.csv"))

metrics_07 %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, color = "train")) +
  geom_line(aes(y = val_loss, color = "val"))

pred_07 <- read_csv(file.path(exp_dir, "runs", "07", "output", "data", "predictions.csv"))

pred_07 %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)

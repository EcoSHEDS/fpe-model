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

set.seed(2137)
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
  mutate(across(c(value, prediction), scale)) %>%
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

cor(pred_04$value, pred_04$prediction, method = "kendall")

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

metrics_05 <- read_csv(file.path(exp_dir, "runs", "05", "output", "data", "metrics.csv"))

metrics_05 %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, color = "train")) +
  geom_line(aes(y = val_loss, color = "val"))

pred_05 <- read_csv(file.path(exp_dir, "runs", "05", "output", "data", "predictions.csv"))

pred_05 %>%
  mutate(across(c(value, prediction), scale)) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)

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
  geom_point(aes(color = split), size = 0.5) +
  # geom_blank(aes(prediction, value)) +
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
  geom_point(aes(color = split), size = 0.5) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)

bind_rows(
  `rank(2000)` = pred_07,
  `reg+rank(2000)` = pred_06,
  .id = "model"
) %>%
  group_by(model) %>%
  mutate(
    across(c(value, prediction), scale)
  ) %>%
  ungroup() %>%
  ggplot(aes(value, prediction)) +
  # geom_abline() +
  geom_point(aes(color = model), size = 0.5) +
  # geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)

bind_rows(
  `rank(2000)` = pred_07,
  `reg+rank(2000)` = pred_06,
  .id = "model"
) %>%
  group_by(model) %>%
  summarize(
    tau = cor(value, prediction, method = "kendall")
  )


bind_rows(
  `rank(500)` = pred_05,
  `rank(2000)` = pred_07,
  `reg+rank(500)` = pred_03,
  `reg+rank(2000)` = pred_06,
  .id = "model"
) %>%
  group_by(model) %>%
  summarize(
    tau = cor(value, prediction, method = "kendall")
  ) %>%
  knitr::kable()

bind_rows(
  `rank(500)` = pred_05,
  `rank(2000)` = pred_07,
  `reg+rank(500)` = pred_03,
  `reg+rank(2000)` = pred_06,
  .id = "model"
) %>%
  group_by(model) %>%
  mutate(
    across(c(value, prediction), scale)
  ) %>%
  ungroup() %>%
  ggplot(aes(value, prediction)) +
  # geom_abline() +
  geom_point(aes(color = model), size = 0.5) +
  # geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)



# run 08 ------------------------------------------------------------------
# same as run 03/06 but with 100 pairs

dir.create(file.path(exp_dir, "runs", "08", "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(exp_dir, "runs", "08", "output", "model"), recursive = TRUE, showWarnings = FALSE)
file.copy(
  file.path(exp_dir, "runs", "03", "input", "model.pth"),
  file.path(exp_dir, "runs", "08", "input", "model.pth")
)
file.copy(
  file.path(exp_dir, "runs", "03", "input", "images.csv"),
  file.path(exp_dir, "runs", "08", "input", "images.csv")
)
n_train <- 100
n_val <- n_train / 0.8 - n_train
set.seed(1118)

pairs_08_train <- pairs_03_all %>%
  filter(split == "train") %>%
  nest_by(split, pair) %>%
  ungroup() %>%
  slice_sample(n = n_train, replace = FALSE) %>%
  unnest(data)
pairs_08_val <- pairs_03_all %>%
  filter(split == "val") %>%
  nest_by(split, pair) %>%
  ungroup() %>%
  slice_sample(n = n_val, replace = FALSE) %>%
  unnest(data)

pairs_08 <- bind_rows(pairs_08_train, pairs_08_val)

summary(pairs_08)

pairs_08 %>%
  write_csv(file.path(exp_dir, "runs", "08", "input", "pairs.csv"))

metrics_08 <- read_csv(file.path(exp_dir, "runs", "08", "output", "data", "metrics.csv"))

metrics_08 %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, color = "train")) +
  geom_line(aes(y = val_loss, color = "val"))

pred_08 <- read_csv(file.path(exp_dir, "runs", "08", "output", "data", "predictions.csv"))

pred_08 %>%
  ggplot(aes(value, prediction)) +
  geom_point(aes(color = split), size = 0.5) +
  # geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)


# run 09 ------------------------------------------------------------------
# train ranknet using run 08 pairs

dir.create(file.path(exp_dir, "runs", "09", "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(exp_dir, "runs", "09", "output", "model"), recursive = TRUE, showWarnings = FALSE)
file.copy(
  file.path(exp_dir, "runs", "08", "input", "pairs.csv"),
  file.path(exp_dir, "runs", "09", "input", "pairs.csv")
)
file.copy(
  file.path(exp_dir, "runs", "08", "input", "images.csv"),
  file.path(exp_dir, "runs", "09", "input", "images.csv")
)

metrics_09 <- read_csv(file.path(exp_dir, "runs", "09", "output", "data", "metrics.csv"))

metrics_09 %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, color = "train")) +
  geom_line(aes(y = val_loss, color = "val"))

pred_09 <- read_csv(file.path(exp_dir, "runs", "09", "output", "data", "predictions.csv"))

pred_09 %>%
  ggplot(aes(value, prediction)) +
  geom_point(aes(color = split), size = 0.5) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)

# run 10 ------------------------------------------------------------------
# same as run 03/06/08 but with 250 pairs

dir.create(file.path(exp_dir, "runs", "10", "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(exp_dir, "runs", "10", "output", "model"), recursive = TRUE, showWarnings = FALSE)
file.copy(
  file.path(exp_dir, "runs", "03", "input", "model.pth"),
  file.path(exp_dir, "runs", "10", "input", "model.pth")
)
file.copy(
  file.path(exp_dir, "runs", "03", "input", "images.csv"),
  file.path(exp_dir, "runs", "10", "input", "images.csv")
)
n_train <- 250
n_val <- floor(n_train / 0.8 - n_train)
set.seed(1118)

pairs_10_train <- pairs_03_all %>%
  filter(split == "train") %>%
  nest_by(split, pair) %>%
  ungroup() %>%
  slice_sample(n = n_train, replace = FALSE) %>%
  unnest(data)
pairs_10_val <- pairs_03_all %>%
  filter(split == "val") %>%
  nest_by(split, pair) %>%
  ungroup() %>%
  slice_sample(n = n_val, replace = FALSE) %>%
  unnest(data)

pairs_10 <- bind_rows(pairs_10_train, pairs_10_val)

summary(pairs_10)

pairs_10 %>%
  write_csv(file.path(exp_dir, "runs", "10", "input", "pairs.csv"))

metrics_10 <- read_csv(file.path(exp_dir, "runs", "10", "output", "data", "metrics.csv"))

metrics_10 %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, color = "train")) +
  geom_line(aes(y = val_loss, color = "val"))

pred_10 <- read_csv(file.path(exp_dir, "runs", "10", "output", "data", "predictions.csv"))

pred_10 %>%
  ggplot(aes(value, prediction)) +
  geom_point(aes(color = split), size = 0.5) +
  # geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)


# run 11 ------------------------------------------------------------------
# train ranknet using run 10 pairs

dir.create(file.path(exp_dir, "runs", "11", "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(exp_dir, "runs", "11", "output", "model"), recursive = TRUE, showWarnings = FALSE)
file.copy(
  file.path(exp_dir, "runs", "10", "input", "pairs.csv"),
  file.path(exp_dir, "runs", "11", "input", "pairs.csv")
)
file.copy(
  file.path(exp_dir, "runs", "10", "input", "images.csv"),
  file.path(exp_dir, "runs", "11", "input", "images.csv")
)

metrics_11 <- read_csv(file.path(exp_dir, "runs", "11", "output", "data", "metrics.csv"))

metrics_11 %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, color = "train")) +
  geom_line(aes(y = val_loss, color = "val"))

pred_11 <- read_csv(file.path(exp_dir, "runs", "11", "output", "data", "predictions.csv"))

pred_11 %>%
  ggplot(aes(value, prediction)) +
  geom_point(aes(color = split), size = 0.5) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)


# run 12 ------------------------------------------------------------------
# run pretrained reg model on prediction table

dir.create(file.path(exp_dir, "runs", "12", "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(exp_dir, "runs", "12", "output", "model"), recursive = TRUE, showWarnings = FALSE)
file.copy(
  file.path(exp_dir, "runs", "11", "input", "images.csv"),
  file.path(exp_dir, "runs", "12", "input", "images.csv")
)
file.copy(
  file.path(exp_dir, "runs", "03", "input", "model.pth"),
  file.path(exp_dir, "runs", "12", "input", "model.pth")
)

pred_12 <- read_csv(file.path(exp_dir, "runs", "12", "output", "data", "predictions.csv"))

pred_12 %>%
  ggplot(aes(value, prediction)) +
  geom_point(aes(color = split), size = 0.5) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)


bind_rows(
  `rank(0)` = pred_04,
  `rank(100)` = pred_09,
  `rank(250)` = pred_11,
  `rank(500)` = pred_05,
  `rank(2000)` = pred_07,
  `reg+rank(0)` = pred_12,
  `reg+rank(100)` = pred_08,
  `reg+rank(250)` = pred_10,
  `reg+rank(500)` = pred_03,
  `reg+rank(2000)` = pred_06,
  .id = "model"
) %>%
  mutate(
    model = fct_inorder(model),
    n_pairs = parse_number(as.character(model)),
    model_type = case_when(
      str_starts(model, "rank") ~ "ranknet only",
      str_starts(model, "reg") ~ "pretrained+ranknet"
    )
  ) %>%
  group_by(n_pairs, model_type) %>%
  summarize(
    tau = cor(value, prediction, method = "kendall")
  ) %>%
  ungroup() %>%
  pivot_wider(names_from = "model_type", values_from = "tau") %>%
  knitr::kable(digits = 3)

bind_rows(
  `rank(0)` = pred_04,
  `rank(100)` = pred_09,
  `rank(250)` = pred_11,
  `rank(500)` = pred_05,
  `rank(2000)` = pred_07,
  `reg+rank(0)` = pred_12,
  `reg+rank(100)` = pred_08,
  `reg+rank(250)` = pred_10,
  `reg+rank(500)` = pred_03,
  `reg+rank(2000)` = pred_06,
  .id = "model"
) %>%
  mutate(model = fct_inorder(model)) %>%
  group_by(model) %>%
  mutate(
    across(c(value, prediction), scale),
    n_pairs = parse_number(as.character(model)),
    model_type = case_when(
      str_starts(model, "rank") ~ "ranknet only",
      str_starts(model, "reg") ~ "pretrained+ranknet"
    )
  ) %>%
  ungroup() %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = factor(n_pairs)), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_grid(vars(model_type), vars(n_pairs)) +
  labs(x = "z(observed log10[flow])", y = "z(predicted score)", color = "# train pairs") +
  theme(aspect.ratio = 1)


bind_rows(
  `rank(0)` = pred_04,
  `rank(100)` = pred_09,
  `rank(250)` = pred_11,
  `rank(500)` = pred_05,
  `rank(2000)` = pred_07,
  `reg+rank(0)` = pred_12,
  `reg+rank(100)` = pred_08,
  `reg+rank(250)` = pred_10,
  `reg+rank(500)` = pred_03,
  `reg+rank(2000)` = pred_06,
  .id = "model"
) %>%
  mutate(model = fct_inorder(model)) %>%
  group_by(model) %>%
  mutate(
    across(c(value, prediction), ~ rank(.) / length(.)),
    n_pairs = parse_number(as.character(model)),
    model_type = case_when(
      str_starts(model, "rank") ~ "ranknet only",
      str_starts(model, "reg") ~ "pretrained+ranknet"
    )
  ) %>%
  ungroup() %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = factor(n_pairs)), size = 0.5, alpha = 0.25) +
  scale_x_continuous(breaks = c(0, 0.5, 1), labels = scales::percent, limits = c(0, 1)) +
  scale_y_continuous(breaks = c(0, 0.5, 1), labels = scales::percent, limits = c(0, 1)) +
  facet_grid(vars(model_type), vars(n_pairs)) +
  labs(x = "rank(observed log10[flow])", y = "rank(predicted score)", color = "# train pairs") +
  theme(aspect.ratio = 1)

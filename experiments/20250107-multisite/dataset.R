# generate dataset for 20250107-multisite

library(tidyverse)
library(jsonlite)
library(janitor)
library(glue)

exp_dir <- "~/git/fpe-model/experiments/20250107-multisite/"
setwd(exp_dir)
img_dir <- "~/data/fpe/images"


# fetch stations ----------------------------------------------------------
# list all stations from API

url <- "https://drekttvuk1.execute-api.us-west-2.amazonaws.com/api/public/stations"
api_stations <- read_json(url, flatten = TRUE, simplifyVector = TRUE) %>%
  as_tibble() %>%
  rowwise() %>%
  mutate(
    n_variables = length(variables),
    has_flow = "FLOW_CFS" %in% variables,
    has_data = (!is.na(nwis_id) & nwis_id != "") | (has_flow),
    has_models = {
      if (nrow(models) == 0) {
        FALSE
      } else {
        models %>%
          filter(variable_id == "FLOW_CFS") %>%
          nrow() > 0
      }
    }
  ) %>%
  ungroup() %>%
  print()

api_stations %>% tabyl(has_models, has_data)
api_stations %>%
  tabyl(has_data)

stations <- api_stations %>%
  filter(
    has_data,
    waterbody_type == "ST",
    str_detect(affiliation_code, "USGS")
  ) %>%
  select(
    affiliation_code, id, name, description, timezone, nwis_id, waterbody_type,
    images.start_date, images.end_date, images.count,
    variables
  )

view(stations)
stations %>%
  tabyl(affiliation_code)

# stations %>%
#   pull(id) %>%
#   write_lines(file.path(exp_dir, "stations", "stations.txt"))

# then run:
# ./batch-station-dataset.sh /mnt/d/fpe/experiments/20250107-multisite/stations/stations.txt /mnt/d/fpe/experiments/20250107-multisite/stations FLOW_CFS


# load all images ---------------------------------------------------------

# daytime only
min_hour <- 8
max_hour <- 16

images <- stations %>%
  rowwise() %>%
  mutate(
    data = list({
      f <- file.path(exp_dir, "stations", id, "data", "images.csv")
      read_csv(f, show_col_types = FALSE) %>%
        mutate(
          timestamp_hour = hour(with_tz(timestamp, tz = timezone)),
          value = log10(pmax(value, 0.001))
        ) %>%
        filter(
          !is.na(value),
          timestamp_hour >= min_hour,
          timestamp_hour <= max_hour
        )
    }),
    n_images = nrow(data)
  ) %>%
  filter(n_images > 1000)

images %>%
  select(data) %>%
  unnest(data) %>%
  tabyl(timestamp_hour)


# base run ------------------------------------------------------------------
# train regression model on obs. flow
# test: last 1 months (n=100)
# train/val: all but last 2 months (n=120)

set.seed(2102)

test_station_ids <- c(140, sample(images$id, size = 10, replace = FALSE)) # include 140-CAMBRIDGE RES., UNNAMED TRIB 2, NR LEXINGTON, MA (short POR)
images_base <- images %>%
  select(affiliation_code, id, name, timezone, data) %>%
  mutate(
    station_split = if_else(id %in% test_station_ids, "test", "train")
  ) %>%
  mutate(
    data = list({
      if (station_split == "test") {
        out <- data %>%
          mutate(split = "test-out") %>%
          slice_sample(n = 650, replace = FALSE)
      } else {
        end <- max(data$timestamp)
        test_start <- floor_date(end - months(1), "day")
        test_out <- data %>%
          filter(timestamp >= test_start) %>%
          slice_sample(n = 200, replace = FALSE)
        train <- data %>%
          filter(timestamp < test_start) %>%
          slice_sample(n = 200, replace = FALSE)
        val <- data %>%
          filter(
            timestamp < test_start,
            !image_id %in% train$image_id
          ) %>%
          slice_sample(n = 50, replace = FALSE)
        test_in <- data %>%
          filter(
            timestamp < test_start,
            !image_id %in% train$image_id,
            !image_id %in% val$image_id
          ) %>%
          slice_sample(n = 200, replace = FALSE)
        out <- bind_rows(
          train = train,
          val = val,
          `test-in` = test_in,
          `test-out` = test_out,
          .id = "split"
        )
      }
      out
    })
  ) %>%
  unnest(data)

images_base %>% distinct(station_split, station_id) %>% tabyl(station_split)

images %>%
  filter(id != 367) %>%
  mutate(
    station_split = if_else(id %in% test_station_ids, "test", "train")
  ) %>%
  arrange(desc(station_split), id) %>%
  left_join(
    api_stations %>%
      select(id, latitude, longitude),
    by = "id"
  ) %>%
  transmute(
    station_split,
    affiliation_code,
    station_code = glue("{id}-{name}"),
    nwis_id,
    latitude,
    longitude,
    images.start_date,
    images.end_date,
    n_days = n_distinct(as_date(data$timestamp)),
    n_images
  ) %>%
  write_csv("figs/stations.csv")

images_base %>%
  tabyl(split, station_split) %>%
  adorn_totals(where = "both")

images_base %>%
  mutate(timestamp_hour2 = map2_int(timestamp, timezone, \(x, y) hour(with_tz(x, tz = y)))) %>%
  tabyl(timestamp_hour, timestamp_hour2)

images_base %>%
  ggplot(aes(timestamp, value)) +
  geom_point(aes(color = split), size = 0.5) +
  facet_wrap(vars(name), scales = "free")

images_base %>%
  ggplot(aes(timestamp_hour, value)) +
  geom_jitter(aes(color = split), size = 0.5, width = 0.25, height = 0) +
  facet_wrap(vars(name), scales = "free")

images_base %>%
  ggplot(aes(value)) +
  stat_ecdf(aes(color = split)) +
  facet_wrap(vars(name), scales = "free")

images_base %>%
  ggplot(aes(value)) +
  stat_ecdf(aes(color = split))

dir.create(file.path(exp_dir, "runs", "base", "input"), recursive = TRUE)
images_base %>%
  write_csv(file.path(exp_dir, "runs", "base", "input", "images.csv"))

images_base %>%
  pull(filename) %>%
  unique() %>%
  toJSON(auto_unbox = TRUE) %>%
  str_replace(
    "\\[", "[{\"prefix\": \"s3://usgs-chs-conte-prod-fpe-storage/\"},"
  ) %>%
  write_file(file.path(exp_dir, "runs", "base", "input", "manifest.json"))

images_base %>%
  pull(filename) %>%
  unique() %>%
  write_lines(file.path(exp_dir, "images", "images-base.txt"))

metrics_base <- read_csv(file.path(exp_dir, "runs", "base", "output", "data", "metrics.csv"))

metrics_base %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, color = "train")) +
  geom_line(aes(y = val_loss, color = "val"))

pred_base <- read_csv(file.path(exp_dir, "runs", "base", "output", "data", "predictions.csv")) %>%
  arrange(desc(station_split), station_name, timestamp) %>%
  select(-station_split) %>%
  left_join(
    images_base %>%
      distinct(id, station_split),
    by = "id"
  ) %>%
  arrange(desc(station_split), id) %>%
  transmute(
    station_split = factor(station_split, levels = c("train", "test")),
    station_id,
    station_code = fct_inorder(glue("{station_id}-{station_name}")),
    split = factor(split, levels = c("train", "val", "test-in", "test-out")),
    image_id,
    timestamp,
    filename,
    value,
    prediction
  ) %>%
  filter(value > -3)
pred_base %>%
  write_rds("cache/pred_base.rds")
pred_base <- read_rds("cache/pred_base.rds")

distinct(pred_base, station_split, station_code) %>% view()

pred_base %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  labs(x = "obs. log10[flow]", y = "pred. log10[flow]", color = "image_split") +
  facet_grid(vars(station_split), vars(image_split = split), labeller = label_both) +
  theme(aspect.ratio = 1)

pred_base %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5) +
  scale_color_brewer(palette = "Set1") +
  scale_x_datetime(date_labels = "%b '%y") +
  labs(x = "timestamp", y = "log10[flow]", color = "image_split") +
  facet_wrap(vars(station_code), scales = "free", labeller = label_wrap_gen()) +
  theme(
    strip.text = element_text(size = 8),
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
  )

pred_base %>%
  group_by(station_code) %>%
  mutate(
    tau = cor(value, prediction, method = "kendall"),
    tau = glue("tau = {sprintf('%.3f', tau)}")
  ) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  labs(x = "obs. log10[flow]", y = "pred. log10[flow]", color = "image_split") +
  facet_wrap(vars(station_code, tau), scales = "free", nrow = 5) +
  theme(
    aspect.ratio = 1,
    strip.text = element_text(size = 8)
  )


pred_base %>%
  group_by(station_split, split) %>%
  summarise(
    tau = cor(value, prediction, method = "kendall")
  ) %>%
  pivot_wider(names_from = "station_split", values_from = "tau") %>%
  knitr::kable(digits = 3)

pred_base %>%
  bind_rows(
    pred_base %>%
      filter(station_split == "train") %>%
      mutate(split = "total")
  ) %>%
  mutate(
    split = factor(split, levels = c(levels(pred_base$split), "total"))
  ) %>%
  group_by(station_split, station_name, split) %>%
  summarise(
    tau = cor(value, prediction, method = "kendall")
  ) %>%
  ggplot(aes(split, tau)) +
  geom_hline(yintercept = 0) +
  geom_boxplot() +
  geom_jitter(aes(color = split), height = 0, width = 0.2, alpha = 0.5) +
  scale_color_brewer(palette = "Set1") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 8), limits = c(-0.5, 1)) +
  labs(x = "image_split", color = "image_split") +
  facet_wrap(vars(station_split), labeller = label_both)

pred_base %>%
  bind_rows(
    pred_base %>%
      filter(station_split == "train") %>%
      mutate(split = "total")
  ) %>%
  mutate(
    split = factor(split, levels = c(levels(pred_base$split), "total"))
  ) %>%
  group_by(station_split, station_name, split) %>%
  summarise(
    rmse = sqrt(mean((value - prediction) ^ 2))
  ) %>%
  ggplot(aes(split, rmse)) +
  geom_hline(yintercept = 0) +
  geom_boxplot() +
  geom_jitter(aes(color = split), height = 0, width = 0.2, alpha = 0.5) +
  scale_color_brewer(palette = "Set1") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 8)) +
  labs(x = "image_split", y = "RMSE (log[flow])", color = "image_split") +
  facet_wrap(vars(station_split), labeller = label_both)

pred_base %>%
  group_by(station_id_name, station_split) %>%
  summarise(
    tau = cor(value, prediction, method = "kendall"),
    rmse = sqrt(mean((value - prediction) ^ 2))
  ) %>%
  ungroup() %>%
  arrange(desc(tau)) %>%
  knitr::kable(digits = 3)

# load pairs --------------------------------------------------
# for each station, load all available pairs

stn_pairs <- images_base %>%
  distinct(affiliation_code, station_id, station_name, station_split, timezone) %>%
  rowwise() %>%
  mutate(
    pairs = list({
      stn_dir <- file.path(exp_dir, "stations", station_id)
      f_pairs <- file.path(stn_dir, "model", "pairs.csv")
      if (!file.exists(f_pairs)) {
        tibble()
      } else {
        read_csv(f_pairs, show_col_types = FALSE) %>%
          mutate(
            timestamp_hour_1 = hour(with_tz(timestamp_1, tz = timezone)),
            timestamp_hour_2 = hour(with_tz(timestamp_2, tz = timezone)),
          ) %>%
          filter(
            timestamp_hour_1 %in% min_hour:max_hour,
            timestamp_hour_2 %in% min_hour:max_hour
          )
      }
    }),
    n_pairs = nrow(pairs)
  ) %>%
  filter(n_pairs > 500)

stn_pairs %>%
  tabyl(station_split) %>%
  adorn_totals()
# n=34 (6 test)


# run 01: base+rank(200) ---------------------------------------------

dir.create(file.path(exp_dir, "runs", "run-01"))

set.seed(1017)
n_train_01 <- 200
n_val_01 <- n_train_01 / 0.8 - n_train_01
stn_01 <- stn_pairs %>%
  rowwise() %>%
  mutate(
    pairs = list({
      train <- pairs %>%
        filter(split == "train") %>%
        nest_by(split, pair) %>%
        ungroup() %>%
        slice_sample(n = n_train_01, replace = FALSE) %>%
        unnest(data)
      val <- pairs %>%
        filter(split == "val") %>%
        nest_by(split, pair) %>%
        ungroup() %>%
        slice_sample(n = n_val_01, replace = FALSE) %>%
        unnest(data)
      bind_rows(train, val)
    })
  )
stn_01 %>%
  unnest(pairs) %>%
  tabyl(station_name, split)

# manifest
stn_01 %>%
  unnest(pairs) %>%
  select(filename_1, filename_2) %>%
  pivot_longer(everything()) %>%
  distinct(value) %>%
  pull(value) %>%
  write_lines(file.path(exp_dir, "images", "images-pairs-200.txt"))

stn_01 %>%
  pull(station_id) %>%
  sort() %>%
  write_lines(file.path(exp_dir, "runs", "run-01", "stations.txt"))
stn_01 %>%
  filter(station_split == "test") %>%
  pull(station_id) %>%
  sort() %>%
  write_lines(file.path(exp_dir, "runs", "run-01", "stations-test.txt"))
stn_01 %>%
  filter(station_split == "train") %>%
  pull(station_id) %>%
  sort() %>%
  write_lines(file.path(exp_dir, "runs", "run-01", "stations-train.txt"))

for (i in 1:nrow(stn_01)) {
  station_id <- stn_01$station_id[[i]]
  i_pairs <- stn_01$pairs[[i]]
  i_images <- images_base %>%
    filter(station_id == !!station_id)

  stn_dir <- file.path(exp_dir, "runs", "run-01", glue("station-{station_id}"))
  dir.create(file.path(stn_dir, "input"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "data"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "model"), showWarnings = FALSE, recursive = TRUE)

  i_pairs %>%
    write_csv(file.path(stn_dir, "input", "pairs.csv"))
  i_images %>%
    write_csv(file.path(stn_dir, "input", "images.csv"))

  file.copy(
    file.path(exp_dir, "runs", "base", "output", "model", "model.pth"),
    file.path(stn_dir, "input")
  )
}

# ./run-train-rank-station-docker.sh --run 01 --stations-file runs/run-01/stations.txt --pretrained

pred_01 <- stn_01 %>%
  mutate(
    data = list({
      f <- file.path(exp_dir, "runs", "run-01", glue("station-{station_id}"), "output", "data", "predictions.csv")
      if (!file.exists(f)) {
        tibble()
      } else {
        read_csv(f, show_col_types = FALSE) %>%
          mutate(
            across(c(name, station_name), as.character)
          )
      }
    }),
    tau = cor(data$value, data$prediction, method = "kendall")
  )

metrics_01 <- stn_01 %>%
  mutate(
    data = list({
      f <- file.path(exp_dir, "runs", "run-01", glue("station-{station_id}"), "output", "data", "metrics.csv")
      if (!file.exists(f)) {
        tibble()
      } else {
        read_csv(f, show_col_types = FALSE)
      }
    })
  )

metrics_01 %>%
  select(station_id, station_name, station_split, data) %>%
  unnest(data) %>%
  group_by(station_id) %>%
  mutate(
    train_loss = train_loss - max(train_loss),
    val_loss = val_loss - max(val_loss)
  ) %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, group = station_id, color = "train")) +
  geom_line(aes(y = val_loss, group = station_id, color = "val"))


# run 02: rank(200) ---------------------------------------------

dir.create(file.path(exp_dir, "runs", "run-02"), showWarnings = FALSE)

for (i in 1:nrow(stn_01)) {
  station_id <- stn_01$station_id[[i]]
  i_pairs <- stn_01$pairs[[i]]
  i_images <- images_base %>%
    filter(station_id == !!station_id)

  stn_dir <- file.path(exp_dir, "runs", "run-02", glue("station-{station_id}"))
  dir.create(file.path(stn_dir, "input"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "data"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "model"), showWarnings = FALSE, recursive = TRUE)

  i_pairs %>%
    write_csv(file.path(stn_dir, "input", "pairs.csv"))
  i_images %>%
    write_csv(file.path(stn_dir, "input", "images.csv"))
}

# ./run-train-rank-station-docker.sh --run 02 --stations-file runs/run-01/stations.txt

pred_02 <- stn_01 %>%
  mutate(
    data = list({
      f <- file.path(exp_dir, "runs", "run-02", glue("station-{station_id}"), "output", "data", "predictions.csv")
      if (!file.exists(f)) {
        tibble()
      } else {
        read_csv(f, show_col_types = FALSE) %>%
          mutate(
            across(c(name, station_name), as.character)
          )
      }
    }),
    tau = {
      if (nrow(data) > 0) {
        cor(data$value, data$prediction, method = "kendall")
      } else {
        NA_real_
      }
    }
  )


# run 03: base+rank(500) pairs with base ---------------------------------------------

dir.create(file.path(exp_dir, "runs", "run-03"))

set.seed(1123)
n_train_03 <- 500
n_val_03 <- n_train_03 / 0.8 - n_train_03
stn_03 <- stn_pairs %>%
  rowwise() %>%
  mutate(
    pairs = list({
      train <- pairs %>%
        filter(split == "train") %>%
        nest_by(split, pair) %>%
        ungroup() %>%
        slice_sample(n = n_train_03, replace = FALSE) %>%
        unnest(data)
      val <- pairs %>%
        filter(split == "val") %>%
        nest_by(split, pair) %>%
        ungroup() %>%
        slice_sample(n = n_val_03, replace = FALSE) %>%
        unnest(data)
      bind_rows(train, val)
    }),
    n_pairs = nrow(pairs)
  ) %>%
  filter(n_pairs == (n_train_03 + n_val_03) * 2)
stn_03 %>%
  unnest(pairs) %>%
  tabyl(station_name, split)
stn_03 %>%
  tabyl(station_split)

# manifest
stn_03 %>%
  unnest(pairs) %>%
  select(filename_1, filename_2) %>%
  pivot_longer(everything()) %>%
  distinct(value) %>%
  pull(value) %>%
  write_lines(file.path(exp_dir, "images", "images-pairs-500.txt"))

stn_03 %>%
  pull(station_id) %>%
  sort() %>%
  write_lines(file.path(exp_dir, "runs", "run-03", "stations.txt"))
stn_03 %>%
  filter(station_split == "test") %>%
  pull(station_id) %>%
  sort() %>%
  write_lines(file.path(exp_dir, "runs", "run-03", "stations-test.txt"))
stn_03 %>%
  filter(station_split == "train") %>%
  pull(station_id) %>%
  sort() %>%
  write_lines(file.path(exp_dir, "runs", "run-03", "stations-train.txt"))

for (i in 1:nrow(stn_03)) {
  station_id <- stn_03$station_id[[i]]
  i_pairs <- stn_03$pairs[[i]]
  i_images <- images_base %>%
    filter(station_id == !!station_id)

  stn_dir <- file.path(exp_dir, "runs", "run-03", glue("station-{station_id}"))
  dir.create(file.path(stn_dir, "input"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "data"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "model"), showWarnings = FALSE, recursive = TRUE)

  i_pairs %>%
    write_csv(file.path(stn_dir, "input", "pairs.csv"))
  i_images %>%
    write_csv(file.path(stn_dir, "input", "images.csv"))

  file.copy(
    file.path(exp_dir, "runs", "base", "output", "model", "model.pth"),
    file.path(stn_dir, "input")
  )
}

# ./run-train-rank-station-docker.sh --run 03 --stations-file runs/run-03/stations.txt --pretrained

pred_03 <- stn_03 %>%
  mutate(
    data = list({
      f <- file.path(exp_dir, "runs", "run-03", glue("station-{station_id}"), "output", "data", "predictions.csv")
      if (!file.exists(f)) {
        tibble()
      } else {
        read_csv(f, show_col_types = FALSE) %>%
          mutate(
            across(c(name, station_name), as.character)
          )
      }
    }),
    tau = {
      if (nrow(data) > 0) {
        cor(data$value, data$prediction, method = "kendall")
      } else {
        NA_real_
      }
    }
  )



# run 04: rank(500) ---------------------------------------------

dir.create(file.path(exp_dir, "runs", "run-04"), showWarnings = FALSE)

for (i in 1:nrow(stn_03)) {
  station_id <- stn_03$station_id[[i]]
  i_pairs <- stn_03$pairs[[i]]
  i_images <- images_base %>%
    filter(station_id == !!station_id)

  stn_dir <- file.path(exp_dir, "runs", "run-04", glue("station-{station_id}"))
  dir.create(file.path(stn_dir, "input"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "data"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "model"), showWarnings = FALSE, recursive = TRUE)

  i_pairs %>%
    write_csv(file.path(stn_dir, "input", "pairs.csv"))
  i_images %>%
    write_csv(file.path(stn_dir, "input", "images.csv"))
}

# ./run-train-rank-station-docker.sh --run 04 --stations-file runs/run-03/stations.txt

pred_04 <- stn_03 %>%
  mutate(
    data = list({
      f <- file.path(exp_dir, "runs", "run-04", glue("station-{station_id}"), "output", "data", "predictions.csv")
      if (!file.exists(f)) {
        tibble()
      } else {
        read_csv(f, show_col_types = FALSE)
      }
    }),
    tau = {
      if (nrow(data) > 0) {
        cor(data$value, data$prediction, method = "kendall")
      } else {
        NA_real_
      }
    }
  )


# run-05: finetune with smaller lr ----------------------------------------
# same as run-01 (n=200), but set lr=0.001 and early_stopping_patience=3

dir.create(file.path(exp_dir, "runs", "run-05"), showWarnings = FALSE)

for (i in 1:nrow(stn_01)) {
  station_id <- stn_01$station_id[[i]]

  stn_dir <- file.path(exp_dir, "runs", "run-05", glue("station-{station_id}"))
  dir.create(file.path(stn_dir, "input"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "data"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "model"), showWarnings = FALSE, recursive = TRUE)

  file.copy(
    file.path(exp_dir, "runs", "run-01", glue("station-{station_id}"), "input", "pairs.csv"),
    file.path(stn_dir, "input")
  )
  file.copy(
    file.path(exp_dir, "runs", "run-01", glue("station-{station_id}"), "input", "images.csv"),
    file.path(stn_dir, "input")
  )
  file.copy(
    file.path(exp_dir, "runs", "run-01", glue("station-{station_id}"), "input", "model.pth"),
    file.path(stn_dir, "input")
  )
}

# finetune lr=0.01
pred_01_train %>%
  filter(station_id == 93)
# finetune lr=0.001
pred_05 <- stn_01 %>%
  filter(station_id == 93) %>%
  mutate(
    data = list({
      f <- file.path(exp_dir, "runs", "run-05", glue("station-{station_id}"), "output", "data", "predictions.csv")
      if (!file.exists(f)) {
        tibble()
      } else {
        read_csv(f, show_col_types = FALSE)
      }
    }),
    tau = cor(data$value, data$prediction, method = "kendall")
  ) %>%
  print()
# base
pred_base %>%
  filter(station_id == 93) %>%
  group_by(station_name) %>%
  summarise(
    tau = cor(value, prediction, method = "kendall")
  )

# run 06: base+rank(50-500, 5x) ---------------------------------------------

dir.create(file.path(exp_dir, "runs", "run-03"))

set.seed(738)
n_train_06 <- c(50, 100, 250, 500)
n_val_06 <- 100
stn_06 <- stn_pairs %>%
  filter(station_split == "test", station_id %in% c(12, 16, 46, 166)) %>%
  mutate(
    val_pairs = list({
      pairs %>%
        filter(split == "val") %>%
        nest_by(split, pair) %>%
        ungroup() %>%
        slice_sample(n = n_val_06, replace = FALSE) %>%
        unnest(data)
    })
  ) %>%
  crossing(
    n_train = n_train_06,
    trial = 1:5
  ) %>%
  rowwise() %>%
  mutate(
    pairs = list({
      train <- pairs %>%
        filter(split == "train") %>%
        nest_by(split, pair) %>%
        ungroup() %>%
        slice_sample(n = n_train, replace = FALSE) %>%
        unnest(data)
      bind_rows(train, val_pairs)
    }),
    n_pairs = nrow(pairs)
  ) %>%
  select(-val_pairs)
stn_06_val <- stn_06 %>%
  select(station_id, pairs) %>%
  mutate(pairs = list({
    filter(pairs, split == "val")
  })) %>%
  distinct() %>%
  rename(val_pairs = pairs)
stn_06_200 <- stn_pairs %>%
  filter(station_split == "test", station_id %in% c(12, 16, 46, 166)) %>%
  left_join(stn_06_val, by = "station_id") %>%
  crossing(
    n_train = 200,
    trial = 1:5
  ) %>%
  rowwise() %>%
  mutate(
    pairs = list({
      train <- pairs %>%
        filter(split == "train") %>%
        nest_by(split, pair) %>%
        ungroup() %>%
        slice_sample(n = n_train, replace = FALSE) %>%
        unnest(data)
      bind_rows(train, val_pairs)
    }),
    n_pairs = nrow(pairs)
  ) %>%
  select(-val_pairs)
  # filter(n_pairs == (n_train_06 + n_val_06) * 2)
stn_06 %>%
  unnest(pairs) %>%
  tabyl(station_name, split, n_train)
stn_06 %>%
  tabyl(n_pairs)

# manifest
stn_06 %>%
  unnest(pairs) %>%
  select(filename_1, filename_2) %>%
  pivot_longer(everything()) %>%
  distinct(value) %>%
  pull(value) %>%
  write_lines(file.path(exp_dir, "images", "images-pairs-run-06.txt"))

stn_06_200 %>%
  unnest(pairs) %>%
  select(filename_1, filename_2) %>%
  pivot_longer(everything()) %>%
  distinct(value) %>%
  pull(value) %>%
  write_lines(file.path(exp_dir, "images", "images-pairs-run-06-200.txt"))

stn_06 %>%
  pull(station_id) %>%
  sort() %>%
  write_lines(file.path(exp_dir, "runs", "run-06", "stations.txt"))
stn_06 %>%
  filter(station_split == "test") %>%
  pull(station_id) %>%
  sort() %>%
  write_lines(file.path(exp_dir, "runs", "run-06", "stations-test.txt"))
stn_06 %>%
  filter(station_split == "train") %>%
  pull(station_id) %>%
  sort() %>%
  write_lines(file.path(exp_dir, "runs", "run-06", "stations-train.txt"))

for (i in 1:nrow(stn_06)) {
  station_id <- stn_06$station_id[[i]]
  i_pairs <- stn_06$pairs[[i]]
  i_images <- images_base %>%
    filter(station_id == !!station_id)

  run_name <- glue("n{stn_06$n_train[[i]]}_t{stn_06$trial[[i]]}_s{stn_06$station_id[[i]]}")
  print(run_name)
  stn_dir <- file.path(exp_dir, "runs", "run-06", run_name)
  dir.create(file.path(stn_dir, "input"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "data"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "model"), showWarnings = FALSE, recursive = TRUE)

  i_pairs %>%
    write_csv(file.path(stn_dir, "input", "pairs.csv"))
  i_images %>%
    write_csv(file.path(stn_dir, "input", "images.csv"))

  file.copy(
    file.path(exp_dir, "runs", "base", "output", "model", "model.pth"),
    file.path(stn_dir, "input")
  )
}
for (i in 1:nrow(stn_06_200)) {
  station_id <- stn_06_200$station_id[[i]]
  i_pairs <- stn_06_200$pairs[[i]]
  i_images <- images_base %>%
    filter(station_id == !!station_id)

  run_name <- glue("n{stn_06_200$n_train[[i]]}_t{stn_06_200$trial[[i]]}_s{stn_06_200$station_id[[i]]}")
  print(run_name)
  stn_dir <- file.path(exp_dir, "runs", "run-06", run_name)
  dir.create(file.path(stn_dir, "input"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "data"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(stn_dir, "output", "model"), showWarnings = FALSE, recursive = TRUE)

  i_pairs %>%
    write_csv(file.path(stn_dir, "input", "pairs.csv"))
  i_images %>%
    write_csv(file.path(stn_dir, "input", "images.csv"))

  file.copy(
    file.path(exp_dir, "runs", "base", "output", "model", "model.pth"),
    file.path(stn_dir, "input")
  )
}
# ./run-train-rank-station-docker.sh --run 03 --stations-file runs/run-03/stations.txt --pretrained

pred_06 <- stn_06 %>%
  filter(n_train != 250) %>%
  bind_rows(stn_06_200) %>%
  mutate(
    run_name = glue("n{n_train}_t{trial}_s{station_id}"),
    station_code = glue("{station_id}-{station_name}"),
    data = list({
      f <- file.path(exp_dir, "runs", "run-06", run_name, "output", "data", "predictions.csv")
      if (!file.exists(f)) {
        tibble()
      } else {
        read_csv(f, show_col_types = FALSE) %>%
          mutate(
            across(c(name, station_name), as.character)
          ) %>%
          filter(value > -3)
      }
    }),
    tau = {
      if (nrow(data) > 0) {
        cor(data$value, data$prediction, method = "kendall")
      } else {
        NA_real_
      }
    },
    rmse = {
      if (nrow(data) > 0) {
        sqrt(mean((scale(data$value) - scale(data$prediction))^2))
      } else {
        NA_real_
      }
    }
  )

x <- pred_06 %>%
  filter(!is.na(tau), station_code != "18-Obear Brook Lower_01171070") %>%
  mutate(
    model = glue("base+rank({n_train})")
  ) %>%
  bind_rows(
    pred %>%
      select(-data) %>%
      unnest(tau) %>%
      filter(station_split == "test", split == "test-out", station_code != "18-Obear Brook Lower_01171070") %>%
      mutate(
        station_code = as.character(station_code),
        model = as.character(model)
      )
  )
x %>%
  filter(model != "base") %>%
  mutate(
    n_pair = parse_number(model),
    model = case_when(
      str_starts(model, "base") ~ "base+rank",
      str_starts(model, "rank") ~ "rank",
      TRUE ~ model
    )
  ) %>%
  filter(!is.na(tau)) %>%
  ggplot(aes(factor(n_pair), tau)) +
  geom_hline(
    data = filter(x, model == "base", !is.na(tau)),
    aes(yintercept = tau, alpha = 0.5, linetype = "dashed")
  ) +
  geom_point(aes(color = model), size = 3, alpha = 0.5) +
  ylim(0.5, 1) +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(station_code)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))

pred_06 %>%
  filter(!is.na(rmse)) %>%
  select(trial, n_train, station_code, tau, rmse) %>%
  pivot_longer(c(tau, rmse)) %>%
  ggplot(aes(factor(n_train), value, color = factor(trial))) +
  geom_point(size = 3, alpha = 0.75) +
  scale_color_brewer(palette = "Set1") +
  facet_grid(vars(name), vars(fct_inorder(station_code)), scales = "free_y")

pred_06 %>%
  filter(!is.na(rmse)) %>%
  select(trial, n_train, station_code, tau, rmse) %>%
  ggplot(aes(tau, rmse)) +
  geom_point(aes(color = factor(trial)), size = 3, alpha = 0.75) +
  geom_smooth(method = "lm", se = FALSE) +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(fct_inorder(station_code)))

pred_06 %>%
  select(run_name, trial, n_train, station_code, data) %>%
  unnest(data) %>%
  group_by(run_name, trial, n_train, station_code) %>%
  mutate(
    across(c(value, prediction), scale)
  ) %>%
  ggplot(aes(value, prediction, color = factor(trial))) +
  geom_abline() +
  geom_point(aes(), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  facet_grid(vars(n_train), vars(station_code))

pred_06 %>%
  ggplot(aes(factor(n_train), trial)) +
  geom_tile(aes(fill = tau)) +
  geom_text(aes(label = sprintf("%.2f", tau)), size = 3) +
  scale_fill_distiller(palette = "YlGnBu", limits = c(0, 1), direction = 1) +
  facet_wrap(vars(station_code))

stn_06 %>%
  filter(station_id == 12, n_train != 250) %>%
  unnest(pairs) %>%
  filter(split == "train") %>%
  mutate(
    true_label = case_when(
      value_1 > value_2 ~ 1,
      value_1 < value_2 ~ -1,
      value_1 == value_2 ~ 0
    ),
    correct = case_when(
      label == 0 ~ TRUE,
      TRUE ~ true_label == label
    )
  ) %>%
  arrange(desc(correct)) %>%
  ggplot(aes(log10(value_1), log10(value_2), color = factor(correct))) +
  geom_abline() +
  geom_point(size = 1, alpha = 0.75) +
  scale_color_brewer(palette = "Set1") +
  facet_grid(vars(n_train), vars(trial)) +
  theme(aspect.ratio = 1)


# predictions -----------------------------------------------------------------

pred_runs <- bind_rows(
  `rank(200)` = pred_02,
  `rank(500)` = pred_04,
  `base+rank(200)` = pred_01,
  `base+rank(500)` = pred_03,
  .id = "model"
) %>%
  mutate(
    station_split = if_else(station_id == 140, "test", station_split)
  ) %>%
  transmute(
    model = fct_inorder(model),
    station_split = factor(station_split, levels = c("train", "test")),
    station_code = glue("{station_id}-{station_name}"),
    data = list({
      data %>%
        transmute(
          split = factor(split, levels = levels(pred_base$split)),
          image_id,
          timestamp,
          filename,
          value,
          prediction
        ) %>%
        filter(value > -3)
    })
  )

pred <- bind_rows(
  pred_base %>%
    mutate(
      station_split = if_else(station_id == 140, "test", station_split)
    ) %>%
    filter(station_code %in% pred_runs$station_code) %>%
    nest_by(
      model = "base",
      station_split,
      station_code
    ) %>%
    arrange(station_split, station_code),
  pred_runs %>%
    mutate(station_code = factor(station_code, levels = levels(pred_base$station_code)))
) %>%
  filter(
    !str_starts(station_code, "78-"),
    !str_starts(station_code, "93-")
  ) %>%
  rowwise() %>%
  mutate(
    model = factor(model, levels = c("base", levels(pred_runs$model))),
    tau = list({
      data %>%
        bind_rows(
          data %>%
            mutate(split = "all")
        ) %>%
        ungroup() %>%
        mutate(split = factor(split, levels = c(levels(data$split), "all"))) %>%
        group_by(split) %>%
        summarise(
          tau = cor(value, prediction, method = "kendall")
        )
    })
  )

pred %>%
  write_rds("cache/pred.rds")
pred <- read_rds("cache/pred.rds")


# memo: base --------------------------------------------------------------

pred_base_tau <- bind_rows(
    pred_base,
    pred_base %>%
      mutate(split = "all")
  ) %>%
  mutate(split = factor(split, levels = c(levels(pred_base$split), "all"))) %>%
  filter(value > -3) %>%
  nest_by(station_split, station_code, split) %>%
  mutate(
    tau = cor(data$value, data$prediction, method = "kendall"),
    rmse = sqrt(mean((data$value - data$prediction) ^ 2)),
    r2 = cor(data$value, data$prediction) ^ 2
  )

p <- pred_base %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  labs(x = "obs. log10[flow]", y = "pred. log10[flow]", color = "image_split") +
  facet_grid(vars(station_split), vars(image_split = split), labeller = label_both) +
  theme_bw() +
  theme(aspect.ratio = 1)
ggsave("figs/base-splot-split.png", plot = p, width = 10, height = 5)

pred_base %>%
  bind_rows(
    pred_base %>%
      filter(station_split == "train") %>%
      mutate(split = "all")
  ) %>%
  mutate(split = factor(split, levels = c(levels(pred_base$split), "all"))) %>%
  group_by(station_split, image_split = split) %>%
  summarise(
    n = n(),
    rmse = sqrt(mean((value - prediction) ^ 2)),
    r2 = cor(value, prediction) ^ 2,
    tau = cor(value, prediction, method = "kendall")
  ) %>%
  write_csv("figs/base-stats.csv")

p_tau <- pred_base_tau %>%
  ggplot(aes(split, fct_rev(station_code))) +
  geom_tile(aes(fill = tau)) +
  geom_text(aes(label = sprintf("%.2f", tau)), size = 3) +
  scale_fill_distiller(palette = "YlGnBu", limits = c(0, 1), direction = 1) +
  facet_grid(vars(station_split), scales = "free_y", space = "free_y", labeller = labeller(
    station_split = label_both
  )) +
  labs(y = "station", x = "image split", subtitle = "Kendall's tau", fill = "tau") +
  theme_bw() +
  theme(
    # aspect.ratio = 1,
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    # strip.text.y = element_text(size = 6)
  )
p_tau

p_rmse <- pred_base_tau %>%
  ggplot(aes(split, fct_rev(station_code))) +
  geom_tile(aes(fill = rmse)) +
  geom_text(aes(label = sprintf("%.2f", rmse)), size = 3) +
  scale_fill_distiller(palette = "YlGnBu", direction = -1, limits = c(0, 1)) +
  facet_grid(vars(station_split), scales = "free_y", space = "free_y", labeller = labeller(
    station_split = label_both
  )) +
  labs(y = "station", x = "image split", subtitle = "RMSE", fill = "RMSE\n(log10[cfs])") +
  theme_bw() +
  theme(
    # aspect.ratio = 1,
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    # strip.text.y = element_text(size = 6)
  )

p_r2 <- pred_base_tau %>%
  ggplot(aes(split, fct_rev(station_code))) +
  geom_tile(aes(fill = r2)) +
  geom_text(aes(label = sprintf("%.2f", r2)), size = 3) +
  scale_fill_distiller(palette = "YlGnBu", direction = 1, limits = c(0, 1)) +
  facet_grid(vars(station_split), scales = "free_y", space = "free_y", labeller = labeller(
    station_split = label_both
  )) +
  labs(y = "station", x = "image split", subtitle = "R^2", fill = "R^2") +
  theme_bw() +
  theme(
    # aspect.ratio = 1,
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    # strip.text.y = element_text(size = 6)
  )

p <- (p_tau | p_r2 | p_rmse) +
  plot_layout(nrow = 1, guides = "collect", axes = "collect")
ggsave("figs/base-stats-heatmap.png", plot = p, width = 12, height = 12)


p_tau <- pred_base_tau %>%
  filter(!(split == "all" & station_split == "test")) %>%
  ggplot(aes(tau, fct_rev(split))) +
  geom_vline(xintercept = 0, alpha = 0.5) +
  geom_boxplot(aes(fill = split)) +
  scale_fill_brewer("image_split", palette = "Set1") +
  facet_grid(vars(station_split), scales = "free_y", space = "free_y") +
  xlim(NA, 1) +
  labs(x = "tau", y = "image split", subtitle = "Kendall's tau") +
  theme_bw() +
  theme(
    # aspect.ratio = 1,
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    # strip.text.y = element_text(size = 6)
  )
p_r2 <- pred_base_tau %>%
  filter(!(split == "all" & station_split == "test")) %>%
  ggplot(aes(r2, fct_rev(split))) +
  geom_vline(xintercept = 0, alpha = 0.5) +
  geom_boxplot(aes(fill = split)) +
  scale_fill_brewer("image_split", palette = "Set1") +
  facet_grid(vars(station_split), scales = "free_y", space = "free_y") +
  xlim(NA, 1) +
  labs(x = "R^2", y = "image split", subtitle = "R^2") +
  theme_bw() +
  theme(
    # aspect.ratio = 1,
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    # strip.text.y = element_text(size = 6)
  )
p_rmse <- pred_base_tau %>%
  filter(!(split == "all" & station_split == "test")) %>%
  ggplot(aes(rmse, fct_rev(split))) +
  geom_boxplot(aes(fill = split)) +
  scale_fill_brewer("image_split", palette = "Set1") +
  facet_grid(vars(station_split), scales = "free_y", space = "free_y") +
  xlim(0, 1) +
  labs(x = "RMSE (log10[cfs])", y = "image split", subtitle = "RMSE") +
  theme_bw() +
  theme(
    # aspect.ratio = 1,
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    # strip.text.y = element_text(size = 6)
  )

p <- (p_tau | p_r2 | p_rmse) +
  plot_layout(nrow = 1, guides = "collect", axes = "collect")
ggsave("figs/base-stats-box-split.png", plot = p, width = 10, height = 4)


p <- pred_base %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.25) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(breaks = scales::pretty_breaks()) +
  scale_y_continuous(breaks = scales::pretty_breaks()) +
  labs(x = "observed log10(flow_cfs)", y = "predicted log10(flow_cfs)", color = "image split") +
  facet_grid(vars(station_split), vars(image_split = split), labeller = label_both) +
  theme_bw() +
  theme(aspect.ratio = 1)
ggsave("figs/base-splot-split.png", plot = p, width = 10, height = 5)

p <- pred_base %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(breaks = scales::pretty_breaks()) +
  scale_y_continuous(breaks = scales::pretty_breaks()) +
  labs(x = "observed log10(flow_cfs)", y = "predicted log10(flow_cfs)", color = "image split") +
  facet_wrap(vars(station_code), ncol = 10, labeller = labeller(
    station_code = label_wrap_gen(width = 20)
  )) +
  theme_bw() +
  theme(
    aspect.ratio = 1,
    strip.text = element_text(size = 6)
  )
ggsave("figs/base-splot-station.png", plot = p, width = 14, height = 10)

p_29_ts <- pred_base %>%
  filter(station_id == 29) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value, linetype = "obs. flow")) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.75) +
  scale_color_brewer(palette = "Set1") +
  guides(
    linetype = guide_legend(order = 2),
    color = guide_legend(order = 1)
  ) +
  labs(
    x = "timestamp", y = "predicted log10(flow_cfs)",
    color = "image\nsplit", linetype = NULL,
    title = pred_base %>%
      distinct(station_id, station_code) %>%
      filter(station_id == 29) %>%
      pull(station_code)
  ) +
  theme_bw()
p_29_splot <- pred_base %>%
  filter(station_id == 29) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.75) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  labs(
    x = "observed log10(flow_cfs)", y = "predicted log10(flow_cfs)",
    color = "image\nsplit"
  ) +
  theme_bw() +
  theme(
    axis.text.y = element_blank(),
    axis.title.y = element_blank()
    # aspect.ratio = 1
  )

p_29 <- (p_29_ts | p_29_splot) +
  plot_layout(ncol = 2, widths = c(3, 1), guides = "collect", axes = "collect")
ggsave("figs/base-29-ts-splot.png", plot = p_29, width = 10, height = 3)

p_12_ts <- pred_base %>%
  filter(station_id == 12) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value, linetype = "obs. flow")) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.75) +
  scale_color_brewer(palette = "Set1", drop = FALSE) +
  guides(
    linetype = guide_legend(order = 2),
    color = guide_legend(order = 1)
  ) +
  labs(
    x = "timestamp", y = "predicted log10(flow_cfs)",
    title = pred_base %>%
      distinct(station_id, station_code) %>%
      filter(station_id == 12) %>%
      pull(station_code),
    color = "image\nsplit", linetype = NULL
  ) +
  theme_bw()
p_12_splot <- pred_base %>%
  filter(station_id == 12) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.75) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1", drop = FALSE) +
  labs(
    x = "observed log10(flow_cfs)", y = "predicted log10(flow_cfs)",
    color = "image\nsplit"
  ) +
  theme_bw() +
  theme(
    axis.text.y = element_blank(),
    axis.title.y = element_blank()
    # aspect.ratio = 1
  )

p_12 <- (p_12_ts | p_12_splot) +
  plot_layout(ncol = 2, widths = c(3, 1), guides = "collect", axes = "collect")
ggsave("figs/base-12-ts-splot.png", plot = p_12, width = 10, height = 3)

p_166_ts <- pred_base %>%
  filter(station_id == 166) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value, linetype = "obs. flow")) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.75) +
  scale_color_brewer(palette = "Set1", drop = FALSE) +
  guides(
    linetype = guide_legend(order = 2),
    color = guide_legend(order = 1)
  ) +
  labs(
    x = "timestamp", y = "predicted log10(flow_cfs)",
    title = pred_base %>%
      distinct(station_id, station_code) %>%
      filter(station_id == 166) %>%
      pull(station_code),
    color = "image\nsplit", linetype = NULL
  ) +
  theme_bw()
p_166_splot <- pred_base %>%
  filter(station_id == 166) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.75) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1", drop = FALSE) +
  labs(
    x = "observed log10(flow_cfs)", y = "predicted log10(flow_cfs)",
    color = "image\nsplit"
  ) +
  theme_bw() +
  theme(
    axis.text.y = element_blank(),
    axis.title.y = element_blank()
    # aspect.ratio = 1
  )

p_166 <- (p_166_ts | p_166_splot) +
  plot_layout(ncol = 2, widths = c(3, 1), guides = "collect", axes = "collect")
ggsave("figs/base-166-ts-splot.png", plot = p_166, width = 10, height = 3)


create_image_grob <- function(image_path, resize_width = 800, ...) {
  img <- load.image(image_path)
  img <- resize(img, resize_width, resize_width * height(img) / width(img))
  grid::rasterGrob(img, interpolate = TRUE, ...)
}

# ts, splot, sample image for 12, 16, 166, 225
set.seed(2149)
p_ts_splot_img <- pred_base %>%
  left_join(
    stations %>%
      select(station_id = id, timezone),
    by = "station_id"
  ) %>%
  filter(station_split == "test") %>%
  nest_by(station_code) %>%
  mutate(
    tau = cor(data$value, data$prediction, method = "kendall"),
    r2 = cor(data$value, data$prediction) ^ 2,
    rmse = sqrt(mean((data$value - data$prediction) ^ 2)),
    p_ts = list({
      data %>%
        ggplot(aes(timestamp)) +
        geom_line(aes(y = value, linetype = "obs. flow")) +
        geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.75) +
        scale_color_brewer(palette = "Set1", drop = FALSE) +
        guides(
          linetype = guide_legend(order = 2),
          color = guide_legend(order = 1)
        ) +
        labs(
          x = "timestamp", y = "predicted log10(flow_cfs)",
          color = "image\nsplit", linetype = NULL,
          title = station_code,
          subtitle = glue("tau = {sprintf('%.3f', tau)} | R^2 = {sprintf('%.3f', r2)} | RMSE = {sprintf('%.2f', rmse)} log10[cfs]")
        ) +
        theme_bw()
    }),
    p_splot = list({
      data %>%
        ggplot(aes(value, prediction)) +
        geom_abline() +
        geom_point(aes(color = split), size = 0.5, alpha = 0.75) +
        geom_blank(aes(prediction, value)) +
        scale_color_brewer(palette = "Set1", drop = FALSE) +
        labs(
          x = "observed log10(flow_cfs)", y = "predicted log10(flow_cfs)",
          color = "image\nsplit"
        ) +
        theme_bw() +
        theme(
          axis.text.y = element_blank(),
          axis.title.y = element_blank()
        )
    }),
    p_img = list({
      f <- data %>%
        filter(
          hour(with_tz(timestamp, tz = timezone)) == 12,
          file.exists(file.path("~/data/fpe/images", filename))
        ) %>%
        slice_sample(n = 1) %>%
        pull(filename)
      create_image_grob(file.path("~/data/fpe/images", f))
    }),
    p = list({
      p_ts_splot <- (p_ts | p_splot) +
        plot_layout(nrow = 1, widths = c(3, 1), guides = "collect")
      wrap_plots(p_ts_splot, p_img) +
        plot_layout(nrow = 1, widths = c(5, 2)) +
        plot_annotation(
          title = station_code,
          subtitle = glue("tau = {sprintf('%.3f', tau)} | R^2 = {sprintf('%.3f', r2)} | RMSE = {sprintf('%.2f', rmse)} log10[cfs]")
        )
    })
  ) %>%
  arrange(desc(r2))
p_ts_splot_img$p[[1]]

ggsave(
  "figs/base-test-ts-splot-a.png",
  wrap_plots(p_ts_splot_img$p[1:6], ncol = 1),
  width = 14,
  height = 18
)
ggsave(
  "figs/base-test-ts-splot-b.png",
  wrap_plots(p_ts_splot_img$p[7:11], ncol = 1),
  width = 14,
  height = 15
)

set.seed(2235)
pdf_base_plots <- pred_base %>%
  nest_by(station_split, station_code) %>%
  left_join(
    pred_base_tau %>%
      ungroup() %>%
      select(-data, -station_split) %>%
      select(station_code, split, tau, `R^2` = r2, `RMSE` = rmse) %>%
      nest_by(station_code, .key = "tau"),
    by = "station_code"
  ) %>%
  mutate(
    plot = list({
      tbl <- gt(tau) %>%
        fmt_number(decimals = 3) %>%
        cols_label(split = "image_split") %>%
        cols_align("left", columns = "split")
      p_ts <- data %>%
        ggplot(aes(timestamp)) +
        geom_line(aes(y = value, linetype = "obs. flow")) +
        geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.75) +
        scale_color_brewer(palette = "Set1", drop = FALSE) +
        guides(
          linetype = guide_legend(order = 2),
          color = guide_legend(order = 1)
        ) +
        labs(
          x = "timestamp", y = "predicted log10(flow_cfs)",
          title = station_code,
          subtitle = glue("station_split: {station_split}"),
          color = "image\nsplit", linetype = NULL
        ) +
        theme_bw()
      p_splot <- data %>%
        ggplot(aes(value, prediction)) +
        geom_abline() +
        geom_point(aes(color = split), size = 0.5, alpha = 0.75) +
        geom_blank(aes(prediction, value)) +
        scale_color_brewer(palette = "Set1", drop = FALSE) +
        labs(
          x = "observed log10(flow_cfs)", y = "predicted log10(flow_cfs)",
          color = "image\nsplit"
        ) +
        theme_bw() +
        theme(
          axis.text.y = element_blank(),
          axis.title.y = element_blank()
          # aspect.ratio = 1
        )

      f <- data %>%
        filter(
          hour(with_tz(timestamp, tz = timezone)) == 12,
          month(timestamp) %in% 3:11,
          file.exists(file.path("~/data/fpe/images", filename))
        ) %>%
        slice_sample(n = 1) %>%
        pull(filename)

      p_img <- create_image_grob(file.path("~/data/fpe/images", f), just = c("center", "top"), y = unit(1, "npc"))

      p1 <- wrap_plots(
        p_ts,
        p_splot,
        ncol = 2,
        widths = c(3, 1),
        guides = "collect",
        axes = "collect"
      )
      p2 <- wrap_plots(
        p_img,
        wrap_table(tbl),
        ncol = 2,
        widths = c(3, 1)
      )
      wrap_plots(p1, p2, ncol = 1, heights = c(1, 2))
    })
  )


pdf("figs/base-diagnostics.pdf", width = 11, height = 8.5)
for (i in 1:nrow(pdf_base_plots)) {
  print(pdf_base_plots$plot[[i]])
}
dev.off()

set_seed(1147)
p_images_train <- images_base %>%
  filter(timestamp_hour == 12, month(timestamp) %in% 3:11) %>%
  filter(file.exists(file.path("~/data/fpe/images", filename))) %>%
  group_by(id) %>%
  slice_sample(n = 1) %>%
  ungroup() %>%
  filter(station_split == "train") %>%
  arrange(station_id) %>%
  transmute(
    station_code = fct_inorder(glue("{station_id}-{station_name}")),
    filename = file.path("~/data/fpe/images", filename)
  ) %>%
  ggplot(aes(0, 0)) +
  geom_image(aes(image = filename), size = 1.2) +
  facet_wrap(vars(station_code), ncol = 6, labeller = labeller(station_code = label_wrap_gen(width = 20))) +
  theme_void() +
  theme(
    panel.border = element_rect(colour = "black"),
    panel.spacing.y = unit(1, "lines"),
    strip.text = element_text(size = 6, face = "bold")
  )
ggsave("figs/station-images-train.png", plot = p_images_train, width = 12, height = 16)

p_images_test <- images_base %>%
  filter(timestamp_hour == 12, month(timestamp) %in% 3:11) %>%
  filter(file.exists(file.path("~/data/fpe/images", filename))) %>%
  group_by(id) %>%
  slice_sample(n = 1) %>%
  ungroup() %>%
  filter(station_split == "test") %>%
  arrange(id) %>%
  transmute(
    station_code = fct_inorder(glue("{station_id}-{station_name}")),
    filename = file.path("~/data/fpe/images", filename)
  ) %>%
  ggplot(aes(0, 0)) +
  geom_image(aes(image = filename), size = 1.2) +
  facet_wrap(vars(station_code), ncol = 3, labeller = labeller(station_code = label_wrap_gen(width = 20))) +
  theme_void() +
  theme(
    panel.border = element_rect(colour = "black"),
    panel.spacing.y = unit(1, "lines"),
    strip.text = element_text(size = 8, face = "bold")
  )
ggsave("figs/station-images-test.png", plot = p_images_test, width = 12, height = 12)


# memo: fine-tune --------------------------------------------------------------------

p <- pred %>%
  unnest(tau) %>%
  ggplot(aes(fct_rev(model), tau)) +
  geom_hline(yintercept = 0, alpha = 0.5) +
  geom_boxplot(aes(fill = station_split)) +
  facet_grid(vars(station_split = fct_rev(station_split)), vars(image_split = split), labeller = label_both) +
  labs(x = "model", y = "tau") +
  scale_fill_brewer(palette = "Set1") +
  coord_flip() +
  ylim(NA, 1) +
  guides(fill = guide_legend(reverse = TRUE)) +
  theme_bw()
ggsave("figs/ft-box-split.png", plot = p, width = 10, height = 4)

p <- pred %>%
  unnest(tau) %>%
  ggplot(aes(model, fct_rev(station_code))) +
  geom_tile(aes(fill = tau)) +
  geom_text(aes(label = sprintf("%.2f", tau)), size = 2) +
  scale_fill_distiller(palette = "YlGnBu", limits = c(0, 1), direction = 1) +
  facet_grid(vars(station_split = fct_rev(station_split)), vars(split), scales = "free_y", space = "free_y", labeller = labeller(station_code = label_wrap_gen())) +
  labs(x = "model", y = "station") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    strip.text.y = element_text(size = 6)
  )
ggsave("figs/ft-heatmap-stations.png", plot = p, width = 12, height = 10)

# fraction of training stations where base is best for all split
pred %>%
  filter(station_split == "train") %>%
  select(-data) %>%
  unnest(tau) %>%
  filter(split == "all") %>%
  group_by(station_code) %>%
  mutate(max_tau = max(tau)) %>%
  pivot_wider(names_from = "model", values_from = "tau") %>%
  mutate(base_is_best = base == max_tau) %>%
  pull(base_is_best) %>%
  mean()

pred %>%
  filter(station_split == "test") %>%
  select(-data) %>%
  unnest(tau) %>%
  filter(split == "all") %>%
  group_by(station_code) %>%
  summarise(
    min_tau = min(tau),
    max_tau = max(tau),
    diff_tau = max_tau - min_tau
  )

pred %>%
  filter(station_split == "test") %>%
  unnest(tau) %>%
  filter(split == "test-out") %>%
  mutate(
    station_code = fct_drop(station_code),
    split = fct_drop(split)
  ) %>%
  complete(station_code, model, split) %>%
  ggplot(aes(fct_rev(station_code), tau)) +
  geom_col(aes(fill = fct_rev(model)), position = "dodge") +
  scale_fill_brewer(palette = "Set1", direction = -1) +
  coord_flip() +
  # geom_tile(aes(fill = tau)) +
  # geom_text(aes(label = sprintf("%.2f", tau)), size = 2) +
  # scale_fill_distiller(palette = "YlGnBu", limits = c(0, 1), direction = 1) +
  facet_wrap(vars(split)) +
  labs(x = "station", y = "tau", fill = "model") +
  guides(fill = guide_legend(reverse = TRUE)) +
  ylim(0, 1) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    strip.text.y = element_text(size = 6)
  )

p <- pred %>%
  filter(station_split == "test") %>%
  unnest(data) %>%
  group_by(station_code, model) %>%
  mutate(across(c(value, prediction), scale)) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value, linetype = "z(obs. flow)")) +
  geom_point(aes(y = prediction, color = split), alpha = 0.5, size = 0.5) +
  scale_x_datetime(date_label = "%b '%y") +
  scale_color_brewer(palette = "Set1", drop = FALSE) +
  guides(
    color = guide_legend(order = 1),
    linetype = guide_legend(order = 2),
  ) +
  facet_grid(vars(model), vars(station_code), scales = "free_x", labeller = label_wrap_gen()) +
  labs(x = "date", y = "z(predicted score)", linetype = NULL, color = "image\nsplit") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    strip.text.x = element_text(size = 6)
  )
ggsave("figs/ft-test-z-ts.png", plot = p, width = 14, height = 9)

p <- pred %>%
  filter(station_split == "test") %>%
  unnest(data) %>%
  group_by(station_code, model) %>%
  mutate(across(c(value, prediction), scale)) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), alpha = 0.5, size = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1", drop = FALSE) +
  guides(
    color = guide_legend(order = 1),
    linetype = guide_legend(order = 2),
  ) +
  facet_grid(vars(model), vars(station_code), labeller = label_wrap_gen()) +
  labs(x = "z(observed flow)", y = "z(predicted score)", linetype = NULL, color = "image\nsplit") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    strip.text.x = element_text(size = 6)
  )
ggsave("figs/ft-test-z-splot.png", plot = p, width = 12, height = 10)

p <- pred %>%
  filter(station_split == "test") %>%
  unnest(data) %>%
  group_by(station_code, model) %>%
  mutate(across(c(value, prediction), ~ (rank(.) - 1) / (n() - 1))) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value, linetype = "z(obs. log[flow_cfs])")) +
  geom_point(aes(y = prediction, color = split), alpha = 0.5, size = 0.5) +
  scale_x_datetime(date_label = "%b '%y") +
  scale_y_continuous(labels = scales::percent) +
  scale_color_brewer(palette = "Set1", drop = FALSE) +
  guides(
    color = guide_legend(order = 1),
    linetype = guide_legend(order = 2),
  ) +
  facet_grid(vars(model), vars(station_code), scales = "free_x", labeller = label_wrap_gen()) +
  labs(x = "date", y = "rank(predicted score)", linetype = NULL, color = "image\nsplit") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    strip.text.x = element_text(size = 6)
  )
ggsave("figs/ft-test-rank-ts.png", plot = p, width = 14, height = 9)

p <- pred %>%
  filter(station_split == "test") %>%
  unnest(data) %>%
  group_by(station_code, model) %>%
  mutate(across(c(value, prediction), ~ (rank(.) - 1) / (n() - 1))) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), alpha = 0.5, size = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1", drop = FALSE) +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent) +
  guides(
    color = guide_legend(order = 1),
    linetype = guide_legend(order = 2),
  ) +
  facet_grid(vars(model), vars(station_code), labeller = label_wrap_gen()) +
  labs(x = "rank(observed log10[flow])", y = "rank(predicted score)", linetype = NULL, color = "image\nsplit") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    strip.text.x = element_text(size = 6)
  )
ggsave("figs/ft-test-rank-splot.png", plot = p, width = 12, height = 10)


# memo: fine tune pdf -----------------------------------------------------

sample_imgs <- pred %>%
  nest_by(station_split, station_code)

pdf_plots <- pred %>%
  mutate(station_id = map_int(as.character(station_code), ~ as.integer(str_split_1(., "-")[[1]]))) %>%
  left_join(
    images_base %>%
      distinct(
        station_id = id,
        timezone
      ),
    by = "station_id"
  ) %>%
  select(-station_id) %>%
  nest_by(station_split, station_code = fct_inorder(station_code)) %>%
  mutate(
    plot_img = list({
      f <- data %>%
        unnest(data) %>%
        filter(
          hour(with_tz(timestamp, tz = timezone)) == 12,
          month(timestamp) %in% 3:11,
          file.exists(file.path("~/data/fpe/images", filename))
        ) %>%
        slice_sample(n = 1) %>%
        pull(filename)

      if (length(f) == 0) {
        f <- data %>%
          unnest(data) %>%
          filter(
            file.exists(file.path("~/data/fpe/images", filename))
          ) %>%
          slice_sample(n = 1) %>%
          pull(filename)
      }

      create_image_grob(
        file.path("~/data/fpe/images", f),
        resize_width = 400,
        just = c("center", "top"),
        y = unit(1, "npc")
      )
    }),
    plot_z = list({
      x_data <- data %>%
        select(model, data) %>%
        unnest(data) %>%
        group_by(model) %>%
        mutate(
          across(c(value, prediction), scale)
        ) %>%
        ungroup()
      x_tau <- data %>%
        select(model, tau) %>%
        unnest(tau) %>%
        pivot_wider(names_from = "split", values_from = "tau")
      if (station_split == "test") {
        x_tau <- x_tau %>%
          select(-all)
      }
      tbl <- gt(x_tau) %>%
        fmt_number(decimals = 3) %>%
        cols_align("left", columns = "model") %>%
        data_color(columns = -model, palette = "YlGnBu", domain = c(0, 1))
      p_ts <- x_data %>%
        ggplot(aes(timestamp)) +
        geom_line(aes(y = value, linetype = "z(obs. flow)")) +
        geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.75) +
        scale_color_brewer(palette = "Set1", drop = FALSE) +
        scale_x_datetime(date_labels = "%b %Y") +
        facet_wrap(vars(model), ncol = 1) +
        guides(
          linetype = guide_legend(order = 2),
          color = guide_legend(order = 1)
        ) +
        labs(
          x = "timestamp", y = "z(predicted score)",
          title = glue("{station_code} | Normalized Values"),
          subtitle = glue("station_split: {station_split}"),
          color = "image\nsplit", linetype = NULL
        ) +
        theme_bw()
      p_splot <- x_data %>%
        ggplot(aes(value, prediction)) +
        geom_abline() +
        geom_point(aes(color = split), size = 0.5, alpha = 0.75) +
        geom_blank(aes(prediction, value)) +
        scale_color_brewer(palette = "Set1", drop = FALSE) +
        facet_wrap(vars(model), ncol = 1) +
        labs(
          x = "z(observed flow)", y = "",
          color = "image\nsplit"
        ) +
        theme_bw() +
        theme(
          axis.text.y = element_blank(),
          axis.title.y = element_blank()
          # aspect.ratio = 1
        )

      p1 <- wrap_plots(
        p_ts,
        p_splot,
        ncol = 2,
        widths = c(7, 1),
        guides = "collect",
        axes = "collect"
      )
      p2 <- wrap_plots(
        plot_img,
        wrap_table(tbl) + ggtitle("Kendall's tau"),
        ncol = 2,
        widths = c(1, 1)
      )
      wrap_plots(p1, p2, ncol = 1, heights = c(3, 1))
    }),
    plot_r = list({
      x_data <- data %>%
        select(model, data) %>%
        unnest(data) %>%
        group_by(model) %>%
        mutate(
          across(c(value, prediction), ~ (rank(.) - 1) / (n() - 1))
        ) %>%
        ungroup()
      x_tau <- data %>%
        select(model, tau) %>%
        unnest(tau) %>%
        pivot_wider(names_from = "split", values_from = "tau")
      if (station_split == "test") {
        x_tau <- x_tau %>%
          select(-all)
      }
      tbl <- gt(x_tau) %>%
        fmt_number(decimals = 3) %>%
        cols_align("left", columns = "model") %>%
        data_color(columns = -model, palette = "YlGnBu", domain = c(0, 1))
      p_ts <- x_data %>%
        ggplot(aes(timestamp)) +
        geom_line(aes(y = value, linetype = "obs. flow")) +
        geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.75) +
        scale_color_brewer(palette = "Set1", drop = FALSE) +
        scale_y_continuous(labels = scales::percent) +
        scale_x_datetime(date_labels = "%b %Y") +
        facet_wrap(vars(model), ncol = 1) +
        guides(
          linetype = guide_legend(order = 2),
          color = guide_legend(order = 1)
        ) +
        labs(
          x = "timestamp", y = "rank(predicted score)",
          title = glue("{station_code} | Rank Percentiles"),
          subtitle = glue("station_split: {station_split}"),
          color = "model", linetype = NULL
        ) +
        theme_bw()
      p_splot <- x_data %>%
        ggplot(aes(value, prediction)) +
        geom_abline() +
        geom_point(aes(color = split), size = 0.5, alpha = 0.75) +
        geom_blank(aes(prediction, value)) +
        scale_x_continuous(labels = scales::percent) +
        scale_y_continuous(labels = scales::percent) +
        scale_color_brewer(palette = "Set1", drop = FALSE) +
        facet_wrap(vars(model), ncol = 1) +
        labs(
          x = "rank(obs. flow)", y = "rank(pred. score)",
          color = "model"
        ) +
        theme_bw() +
        theme(
          axis.text.y = element_blank(),
          axis.title.y = element_blank()
          # aspect.ratio = 1
        )

      p1 <- wrap_plots(
        p_ts,
        p_splot,
        ncol = 2,
        widths = c(7, 1),
        guides = "collect",
        axes = "collect"
      )
      p2 <- wrap_plots(
        plot_img,
        wrap_table(tbl) + ggtitle("Kendall's tau"),
        ncol = 2,
        widths = c(1, 1)
      )
      wrap_plots(p1, p2, ncol = 1, heights = c(3, 1))
    })
  )

pdf("figs/ft-diagnostics.pdf", width = 14, height = 14)
for (i in 1:nrow(pdf_plots)) {
  print(pdf_plots$plot_z[[i]])
  print(pdf_plots$plot_r[[i]])
}
dev.off()



# other plots -------------------------------------------------------------

pred_all %>%
  # filter(station_split == "test") %>%
  select(model, station_id_name_split, data) %>%
  unnest(data) %>%
  group_by(station_id_name_split, model) %>%
  mutate(across(c(value, prediction), scale)) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = model), alpha = 0.5, size = 0.5) +
  scale_x_datetime(date_label = "%b '%y") +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(station_id_name_split), scales = "free", labeller = label_wrap_gen()) +
  labs(x = "date", y = "z(pred. score or obs. log10[flow])") +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    strip.text = element_text(size = 6)
  )

pred_all %>%
  filter(station_split == "test") %>%
  select(model, station_id_name_split, data) %>%
  unnest(data) %>%
  group_by(station_id_name_split, model) %>%
  mutate(across(c(value, prediction), scale)) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = model), alpha = 0.5, size = 0.5) +
  scale_x_datetime(date_label = "%b '%y") +
  scale_color_brewer(palette = "Set1") +
  facet_grid(vars(model), vars(station_id_name_split), scales = "free", labeller = label_wrap_gen()) +
  labs(x = "date", y = "z(pred. score or obs. log10[flow])") +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    strip.text = element_text(size = 6)
  )

pred_all %>%
  # filter(station_split == "test") %>%
  select(model, station_id_name_split, data) %>%
  unnest(data) %>%
  group_by(station_id_name_split, model) %>%
  mutate(across(c(value, prediction), scale)) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = model), alpha = 0.5, size = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(station_id_name_split), scales = "free", nrow = 3, labeller = label_wrap_gen()) +
  labs(x = "z(obs. log10[flow])", y = "z(pred. score)") +
  theme(
    aspect.ratio = 1,
    strip.text = element_text(size = 6)
  )

pred_all %>%
  # filter(station_split == "test") %>%
  select(model, station_id_name_split, data) %>%
  unnest(data) %>%
  group_by(station_id_name_split, model) %>%
  mutate(across(c(value, prediction), ~ rank(.) / length(.))) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = model), alpha = 0.5, size = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(labels = scales::percent, breaks = c(0, 0.5, 1)) +
  scale_y_continuous(labels = scales::percent) +
  facet_wrap(vars(station_id_name_split), scales = "free", nrow = 4, labeller = label_wrap_gen()) +
  labs(x = "rank(obs. log10[flow])", y = "rank(pred. score)") +
  theme(
    axis.text = element_text(size = 4),
    aspect.ratio = 1,
    strip.text = element_text(size = 6)
  )

pred_all_tau %>%
  filter(station_split == "test") %>%
  select(-station_split, -station_id_name_split) %>%
  pivot_wider(names_from = "model", values_from = "tau") %>%
  arrange(desc(`base`)) %>%
  knitr::kable(digits = 3)

pred_all_tau %>%
  filter(station_split == "test") %>%
  ggplot(aes(station_id_name, tau)) +
  geom_point(aes(color = model)) +
  scale_color_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(0, 1)) +
  coord_flip() +
  labs(x = "station", y = "tau") +
  facet_wrap(vars(station_split), labeller = label_both)

pred_all_tau %>%
  ggplot(aes(model, tau)) +
  geom_col(aes(fill = model)) +
  facet_wrap(vars(station_id_name_split), labeller = label_wrap_gen()) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))

pred_all_tau %>%
  ggplot(aes(model, tau)) +
  geom_boxplot() +
  geom_jitter(aes(color = model), width = 0.25, height = 0) +
  scale_y_continuous(limits = c(0, 1), breaks = scales::pretty_breaks(n = 8)) +
  facet_wrap(vars(station_split), labeller = label_both) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))



# stability ---------------------------------------------------------------

# install.packages("imager")  # Uncomment if you haven't installed imager
library(imager)

average_hash <- function(image_path, hash_size=8) {
  tryCatch({
    img <- load.image(image_path)
    gray <- grayscale(img)
    small <- resize(gray, hash_size, hash_size)
    pixels <- as.numeric(small)
    avg_val <- mean(pixels)
    bits <- ifelse(pixels > avg_val, 1, 0)
    return(bits)
  }, error = function(e) {
    warning(sprintf("Failed to process %s: %s", image_path, e$message))
    return(NULL)
  })
}

hamming_distance <- function(hash1, hash2) {
  if (any(is.na(hash1)) || any(is.null(hash1)) || any(is.null(hash2))) return(NA)
  sum(hash1 != hash2) / length(hash1)
}

# Example usage for a sequence of images
x_8 <- images_base %>%
  filter(station_id == 281) %>%
  arrange(timestamp) %>%
  mutate(
    image_hash = map(filename, ~ average_hash(file.path("~/data/fpe/images", .)), .progress = TRUE),
    distance = map2_dbl(lag(image_hash), image_hash, hamming_distance)
  )
x_8 %>%
  ggplot(aes(timestamp, distance)) +
  geom_line() +
  geom_point()


x_16 <- images_base %>%
  filter(station_id == 281) %>%
  arrange(timestamp) %>%
  mutate(
    image_hash = map(filename, ~ average_hash(file.path("~/data/fpe/images", .), hash_size = 16), .progress = TRUE),
    distance = map2_dbl(lag(image_hash), image_hash, hamming_distance)
  )
x_16 %>%
  ggplot(aes(timestamp, distance)) +
  geom_line() +
  geom_point()


library(imager)

compute_ssim <- function(img1_path, img2_path) {
  if (is.na(img1_path) || is.na(img2_path)) return(NA)
  img1 <- load.image(file.path("~/data/fpe/images", img1_path))
  img2 <- load.image(file.path("~/data/fpe/images", img2_path))

  # Convert to grayscale and normalize
  gray1 <- grayscale(img1)
  gray2 <- grayscale(img2)

  # Compute SSIM (using base stats for simplicity)
  # Higher values (closer to 1) indicate more similarity
  mu1 <- mean(gray1)
  mu2 <- mean(gray2)
  sigma1 <- sd(as.vector(gray1))
  sigma2 <- sd(as.vector(gray2))
  sigma12 <- cov(as.vector(gray1), as.vector(gray2))

  # Constants for stability
  c1 <- (0.01)^2
  c2 <- (0.03)^2

  ssim <- ((2*mu1*mu2 + c1)*(2*sigma12 + c2)) /
    ((mu1^2 + mu2^2 + c1)*(sigma1^2 + sigma2^2 + c2))

  return(1 - ssim) # Return dissimilarity score
}

x_ssim_29 <- images_base %>%
  filter(station_id == 29) %>%
  arrange(timestamp) %>%
  mutate(
    ssim = map2_dbl(lag(filename), filename, compute_ssim, .progress = TRUE)
  )
x_ssim_281 <- images_base %>%
  filter(station_id == 281) %>%
  arrange(timestamp) %>%
  mutate(
    ssim = map2_dbl(lag(filename), filename, compute_ssim, .progress = TRUE)
  )

bind_rows(
  x_ssim_29,
  x_ssim_281,
) %>%
  ggplot(aes(timestamp, ssim)) +
  geom_line() +
  geom_point() +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(station_code), ncol = 1, scales = "free_x")

# export images for python
images_base %>%
  filter(station_id == 281) %>%
  arrange(timestamp) %>%
  mutate(
    filename = file.path("/home/jeff/data/fpe/images", filename)
  ) %>%
  pull(filename) %>%
  # unique()
  write_lines("images/base-281.txt")

orb_281 <- jsonlite::fromJSON(read_lines("images/base-281-orb.txt"))
sift_281 <- jsonlite::fromJSON(read_lines("images/base-281-sift.txt"))

images_base %>%
  filter(station_id == 281) %>%
  arrange(timestamp) %>%
  bind_cols(
    orb = c(NA, orb_281),
    sift = c(NA, sift_281)
  ) %>%
  ggplot(aes(timestamp, sift)) +
  geom_line() +
  geom_point(aes(color = orb))
  # scale_color_brewer(palette = "Set1") +
  # facet_wrap(vars(station_code), ncol = 1, scales = "free_x")


images_base %>%
  filter(station_id == 281) %>%
  arrange(timestamp) %>%
  bind_cols(
    orb = c(NA, orb_281),
    sift = c(NA, sift_281)
  ) %>%
  ggplot(aes(orb, sift)) +
  geom_point()

image_files <- c("frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg") # etc.
hashes <- lapply(image_files, average_hash)

# Compute consecutive Hamming distances
distances <- numeric(length(image_files) - 1)
for (i in 1:(length(image_files) - 1)) {
  distances[i] <- hamming_distance(hashes[[i]], hashes[[i+1]])
}

# Now you can examine 'distances' to spot large jumps
print(distances)

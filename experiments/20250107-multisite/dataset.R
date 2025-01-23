# generate dataset for 20250107-multisite

library(tidyverse)
library(jsonlite)
library(janitor)
library(glue)

exp_dir <- "~/git/fpe-model/experiments/20250107-multisite/"
img_dir <- "~/data/fpe/images"


# fetch stations ----------------------------------------------------------
# list all stations from API

url <- "https://drekttvuk1.execute-api.us-west-2.amazonaws.com/api/public/stations"
api_stations <- read_json(url, flatten = TRUE, simplifyVector = TRUE) %>%
  as_tibble() %>%
  mutate(
    n_variables = map_int(variables, length),
    has_flow = map_lgl(variables, \(x) "FLOW_CFS" %in% x),
    has_data = (!is.na(nwis_id) & nwis_id != "") | (has_flow)
  ) %>%
  print()

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

stations %>%
  pull(id) %>%
  write_lines(file.path(exp_dir, "stations", "stations.txt"))

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

test_station_ids <- sample(images$id, size = 10, replace = FALSE)
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
  mutate(
    station_name = fct_inorder(station_name),
    station_split = factor(station_split, levels = c("train", "test")),
    split = factor(split, levels = c("train", "val", "test-in", "test-out"))
  ) %>%
  mutate(
    station_id_name = fct_inorder(glue("{station_id}: {station_name}")),
    station_id_name_split = fct_inorder(glue("{station_id}: {station_name} ({station_split})"))
  )

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
  facet_wrap(vars(station_id_name_split), scales = "free", labeller = label_wrap_gen()) +
  theme(
    strip.text = element_text(size = 8),
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
  )

pred_base %>%
  group_by(station_name) %>%
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
  facet_wrap(vars(station_id_name_split, tau), scales = "free", nrow = 5) +
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


# compare -----------------------------------------------------------------

pred_all <- bind_rows(
  base = pred_base %>%
    filter(station_id %in% pred_02$station_id) %>%
    nest_by(affiliation_code, station_id, station_name, station_split) %>%
    mutate(
      tau = cor(data$value, data$prediction, method = "kendall")
    ),
  `rank(200)` = pred_02,
  `rank(500)` = pred_04,
  `base+rank(200)` = pred_01,
  `base+rank(500)` = pred_03,
  .id = "model"
) %>%
  ungroup() %>%
  rowwise() %>%
  mutate(
    data = list({
      if (nrow(data) == 0) {
        data
      } else {
        data %>%
          select(image_id, timestamp, timestamp_hour, url, value, prediction)
      }
    }),
    model = fct_inorder(model),
    station_split = factor(station_split, levels = c("train", "test"))
  ) %>%
  group_by(station_id) %>%
  mutate(mean_tau = mean(tau, na.rm = TRUE)) %>%
  ungroup() %>%
  arrange(station_split, desc(mean_tau)) %>%
  mutate(
    station_id_name = fct_inorder(glue("{station_id}: {station_name}")),
    station_id_name_split = fct_inorder(glue("{station_id}: {station_name} ({station_split})"))
  )

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


pred_all_tau <- pred_all %>%
  select(model, station_split, station_id_name, station_id_name_split, tau) %>%
  pivot_wider(names_from = "model", values_from = "tau") %>%
  mutate(
    station_id_name = fct_reorder(station_id_name, base)
  ) %>%
  pivot_longer(-c(station_split, station_id_name, station_id_name_split), names_to = "model", values_to = "tau") %>%
  mutate(model = factor(model, levels = levels(pred_all$model)))

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

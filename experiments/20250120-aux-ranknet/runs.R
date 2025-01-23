setwd("~/git/fpe-model/experiments/20250120-aux-ranknet/")

library(tidyverse)
library(janitor)
library(glue)
library(daymetr)
library(openmeteo)

station_id <- 29
daytime_hours <- 8:16

# load: station -----------------------------------------------------------

start_test <- ymd(20240101)

station <- read_json(file.path("stations", station_id, "data", "station.json"))
station_images <- read_csv(file.path("stations", station_id, "data", "images.csv")) %>%
  mutate(
    timestamp_hour = hour(with_tz(timestamp, tzone = station$timezone)),
    value = log10(pmax(value, 0.01))
  ) %>%
  filter(
    !is.na(value),
    timestamp_hour %in% daytime_hours
  )
tabyl(station_images, timestamp_hour)

station_pairs <- read_csv(file.path("stations", station_id, "model", "pairs.csv")) %>%
  filter(
    image_id_1 %in% station_images$image_id,
    image_id_2 %in% station_images$image_id,
    hour(with_tz(timestamp_1, tzone = station$timezone)) %in% daytime_hours,
    hour(with_tz(timestamp_2, tzone = station$timezone)) %in% daytime_hours,
    as_date(with_tz(timestamp_1, tzone = station$timezone)) < start_test,
    as_date(with_tz(timestamp_2, tzone = station$timezone)) < start_test
  ) %>%
  mutate(
    across(starts_with("value_"), ~ log10(pmax(., 0.01)))
  )
summary(station_pairs)

station_pairs %>%
  ggplot(aes(timestamp_1, timestamp_2)) +
  geom_point(aes(color = factor(label))) +
  scale_color_brewer(palette = "Set1")
station_pairs %>%
  ggplot(aes(value_1, value_2)) +
  geom_point(aes(color = factor(label))) +
  scale_color_brewer(palette = "Set1")


# model: 01_ranknet_1000 ----------------------------------------------------------

run_name <- "01_ranknet_1000"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)

n_train_pairs <- 1000
n_val_pairs <- 250 # 80/20 split

set.seed(1952)
run_pairs_train <- station_pairs %>%
  filter(split == "train") %>%
  nest_by(pair) %>%
  ungroup() %>%
  slice_sample(n = n_train_pairs) %>%
  unnest(data)
run_pairs_val <- station_pairs %>%
  filter(split == "val") %>%
  nest_by(pair) %>%
  ungroup() %>%
  slice_sample(n = n_val_pairs) %>%
  unnest(data)

run_pairs <- bind_rows(run_pairs_train, run_pairs_val)

# run_pairs %>%
#   write_csv(file.path(run_station_dir, "input", "pairs.csv"))

run_images <- station_images %>%
  mutate(
    split = case_when(
      as_date(with_tz(timestamp, tzone = station$timezone)) >= start_test ~ "test-out",
      image_id %in% c(run_pairs_train$image_id_1, run_pairs_train$image_id_2) ~ "train",
      image_id %in% c(run_pairs_val$image_id_1, run_pairs_val$image_id_2) ~ "val",
      TRUE ~ "test-in"
    )
  ) %>%
  slice_sample(prop = 0.2)
# run_images %>%
#   write_csv(file.path(run_station_dir, "input", "images.csv"))

tabyl(run_images, split)

run_images %>%
  ggplot(aes(timestamp, value, color = split)) +
  geom_point(size = 0.5) +
  facet_wrap(vars(split))

run_1000_pred <- read_csv(file.path(run_station_dir, "output", "data", "predictions.csv"))

run_1000_pred %>%
  group_by(split) %>%
  summarise(
    n = n(),
    tau = cor(prediction, value, method = "kendall")
  )

run_1000_pred %>%
  mutate(
    across(c(value, prediction), scale)
  ) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  theme(aspect.ratio = 1)

run_1000_pred %>%
  mutate(
    across(c(value, prediction), \(x) (rank(x) - 1) / (length(x) - 1))
  ) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5)


# model: 01_ranknet_500 ----------------------------------------------------------

run_name <- "01_ranknet_500"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)

# use previous run as base
set.seed(2034)
run_pairs <- read_csv(file.path("runs", "01_ranknet_1000", "stations", station_id, "input", "pairs.csv")) %>%
  nest_by(split) %>%
  mutate(
    data = list({
      if (split == "train") {
        data %>%
          nest_by(pair) %>%
          ungroup() %>%
          slice_sample(n = 500) %>%
          unnest(data)
      } else {
        data
      }
    })
  ) %>%
  unnest(data)
tabyl(run_pairs, split)
run_pairs_train <- run_pairs %>%
  filter(split == "train")
run_pairs_val <- run_pairs %>%
  filter(split == "val")

run_pairs %>%
  write_csv(file.path(run_station_dir, "input", "pairs.csv"))

run_images <- read_csv(file.path("runs", "01_ranknet_1000", "stations", station_id, "input", "images.csv")) %>%
  mutate(
    split = case_when(
      image_id %in% c(run_pairs_train$image_id_1, run_pairs_train$image_id_2) ~ "train",
      split == "train" ~ "test-in",
      TRUE ~ split
    )
  )

tabyl(run_images, split)

run_images %>%
  write_csv(file.path(run_station_dir, "input", "images.csv"))

run_images %>%
  ggplot(aes(timestamp, value, color = split)) +
  geom_point(size = 0.5) +
  facet_wrap(vars(split))


run_500_pred <- read_csv(file.path(run_station_dir, "output", "data", "predictions.csv"))

# model: 01_ranknet_200 ----------------------------------------------------------

run_name <- "01_ranknet_200"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)

# use previous run as base
set.seed(2147)
run_pairs <- read_csv(file.path("runs", "01_ranknet_500", "stations", station_id, "input", "pairs.csv")) %>%
  nest_by(split) %>%
  mutate(
    data = list({
      if (split == "train") {
        data %>%
          nest_by(pair) %>%
          ungroup() %>%
          slice_sample(n = 200) %>%
          unnest(data)
      } else {
        data
      }
    })
  ) %>%
  unnest(data)
tabyl(run_pairs, split)
run_pairs_train <- run_pairs %>%
  filter(split == "train")
run_pairs_val <- run_pairs %>%
  filter(split == "val")

run_pairs %>%
  write_csv(file.path(run_station_dir, "input", "pairs.csv"))

run_images <- read_csv(file.path("runs", "01_ranknet_500", "stations", station_id, "input", "images.csv")) %>%
  mutate(
    split = case_when(
      image_id %in% c(run_pairs_train$image_id_1, run_pairs_train$image_id_2) ~ "train",
      split == "train" ~ "test-in",
      TRUE ~ split
    )
  )

tabyl(run_images, split)

run_images %>%
  write_csv(file.path(run_station_dir, "input", "images.csv"))

run_images %>%
  ggplot(aes(timestamp, value, color = split)) +
  geom_point(size = 0.5) +
  facet_wrap(vars(split))

run_200_pred <- read_csv(file.path(run_station_dir, "output", "data", "predictions.csv"))

bind_rows(
  `1000` = run_1000_pred,
  `500` = run_500_pred,
  `200` = run_200_pred,
  .id = "n_train"
) %>%
  group_by(n_train, split) %>%
  summarise(
    n = n(),
    tau = cor(prediction, value, method = "kendall")
  )

bind_rows(
  `1000` = run_1000_pred,
  `500` = run_500_pred,
  `200` = run_200_pred,
  .id = "n_train"
) %>%
  group_by(n_train) %>%
  mutate(
    across(c(value, prediction), scale)
  ) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(n_train)) +
  theme(aspect.ratio = 1)

bind_rows(
  `1000` = run_1000_pred,
  `500` = run_500_pred,
  `200` = run_200_pred,
  .id = "n_train"
) %>%
  group_by(n_train) %>%
  mutate(
    across(c(value, prediction), \(x) (rank(x) - 1) / (length(x) - 1))
  ) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5) +
  facet_wrap(vars(n_train), ncol = 1)


# met: day ----------------------------------------------------------------

met_start <- min(as_date(with_tz(station_images$timestamp, station$timezone))) - days(90)
met_end <- max(as_date(with_tz(station_images$timestamp, station$timezone)))

openmeteo::weather_variables()

met_day <- openmeteo::weather_history(
  location = c(station$latitude, station$longitude),
  start = as.character(met_start),
  end = as.character(met_end),
  daily = c(
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration"
  )
)

met_day %>%
  pivot_longer(-date) %>%
  ggplot(aes(date, value)) +
  geom_line() +
  facet_wrap(vars(name), scales = "free_y")

aux_day_inp <- met_day %>%
  transmute(
    date,
    precip = daily_precipitation_sum,
    airtemp = (daily_temperature_2m_max + daily_temperature_2m_min) / 2,
    srad = daily_shortwave_radiation_sum,
    wspeed = daily_wind_speed_10m_max,
    et = daily_et0_fao_evapotranspiration
  ) %>%
  mutate(
    across(-date, \(x) (x - min(x)) / (max(x) - min(x)))
  )
aux_day_mean <- aux_day_inp %>%
  select(date, airtemp, srad, wspeed, et) %>%
  pivot_longer(-date) %>%
  crossing(
    ndays = c(7, 30, 90)
  ) %>%
  nest_by(name, ndays) %>%
  mutate(
    name = glue("{name}_mean{ndays}d"),
    data = list({
      data %>%
        mutate(
          value = slider::slide_dbl(value, .before = ndays, .after = -1, .f = mean)
        )
    })
  ) %>%
  ungroup() %>%
  select(-ndays) %>%
  unnest(data) %>%
  pivot_wider()
aux_day_sum <- aux_day_inp %>%
  select(date, precip) %>%
  pivot_longer(-date) %>%
  crossing(
    ndays = c(7, 30, 90)
  ) %>%
  nest_by(name, ndays) %>%
  mutate(
    name = glue("{name}_sum{ndays}d"),
    data = list({
      data %>%
        mutate(
          value = slider::slide_dbl(value, .before = ndays, .after = -1, .f = sum)
        )
    })
  ) %>%
  ungroup() %>%
  select(-ndays) %>%
  unnest(data) %>%
  pivot_wider()
aux_day_other <- aux_day_inp %>%
  transmute(
    date,
    precip_lag1d = lag(precip, n = 1),
    precip_lag2d = lag(precip, n = 2),
    precip_lag3d = lag(precip, n = 3),
    precip_lag4d = lag(precip, n = 4),
    precip_lag5d = lag(precip, n = 5),
    precip_lag6d = lag(precip, n = 6),
    precip_lag7d = lag(precip, n = 7),

    precip_max7d = na_if(slider::slide_dbl(precip, .before = 7, .after = -1, .f = max), -Inf),
    precip_max30d = na_if(slider::slide_dbl(precip, .before = 30, .after = -1, .f = max), -Inf),

    precip_sd7d = slider::slide_dbl(precip, .before = 7, .after = -1, .f = sd),
    precip_sd30d = slider::slide_dbl(precip, .before = 30, .after = -1, .f = sd),

    airtemp_lag1d = lag(airtemp, n = 1),
    airtemp_lag2d = lag(airtemp, n = 2),
    airtemp_lag3d = lag(airtemp, n = 3),

    airtemp_max7d = na_if(slider::slide_dbl(airtemp, .before = 7, .after = -1, .f = max), -Inf),
    airtemp_max30d = na_if(slider::slide_dbl(airtemp, .before = 30, .after = -1, .f = max), -Inf),

    airtemp_min7d = na_if(slider::slide_dbl(airtemp, .before = 7, .after = -1, .f = min), Inf),
    airtemp_min30d = na_if(slider::slide_dbl(airtemp, .before = 30, .after = -1, .f = min), Inf),
  )
aux_day <- aux_day_mean %>%
  left_join(aux_day_sum, by = "date") %>%
  left_join(aux_day_other, by = "date") %>%
  mutate(
    across(-date, ~ data.table::nafill(., type = "nocb"))
  ) %>%
  print()
summary(aux_day)

aux_day %>%
  inner_join(
    station_images %>%
      mutate(date = as_date(with_tz(timestamp, tz = station$timezone))) %>%
      group_by(date) %>%
      summarize(flow = mean(value)),
    by = "date"
  ) %>%
  select(-date) %>%
  select(flow, everything()) %>%
  pivot_longer(-flow) %>%
  mutate(name = fct_inorder(name)) %>%
  ggplot(aes(value, flow)) +
  geom_point(size = 0.5, alpha = 0.5) +
  facet_wrap(vars(name), ncol = 4)


# ranknet+scalar(500) -----------------------------------------------------

run_name <- "02_ranknet+scalar_500"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)

# use pairs and images from ranknet(200)
# but add date column for aux lookup
run_pairs <- file.path("runs", "01_ranknet_500", "stations", station_id, "input", "pairs.csv") %>%
  read_csv() %>%
  mutate(
    date_1 = as_date(with_tz(timestamp_1, tzone = station$timezone)),
    date_2 = as_date(with_tz(timestamp_2, tzone = station$timezone))
  )
run_pairs %>%
  write_csv(file.path(run_station_dir, "input", "pairs.csv"))

run_images <- file.path("runs", "01_ranknet_500", "stations", station_id, "input", "images.csv") %>%
  read_csv() %>%
  mutate(
    date = as_date(with_tz(timestamp, tzone = station$timezone))
  )
run_images %>%
  write_csv(file.path(run_station_dir, "input", "images.csv"))

aux_day %>%
  write_csv(file.path(run_station_dir, "input", "aux.csv"))

run_pred_lr01 <- read_csv(file.path("runs", "02_ranknet+scalar_500_lr01", "stations", station_id, "output", "data", "predictions.csv"))
run_pred_lr001 <- read_csv(file.path(run_station_dir, "output", "data", "predictions.csv"))

run_pred_lr01 %>%
  mutate(
    across(c(value, prediction), scale)
  ) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  theme(aspect.ratio = 1)

bind_rows(
  `ranknet(lr=0.001)` = run_500_pred,
  `ranknet+aux(lr=0.01)` = run_pred_lr01,
  `ranknet+aux(lr=0.001)` = run_pred_lr001,
  .id = "model"
) %>%
  group_by(model, split) %>%
  summarise(
    # n = n(),
    tau = cor(prediction, value, method = "kendall")
  ) %>%
  pivot_wider(names_from = split, values_from = tau)

bind_rows(
  `ranknet(lr=0.001)` = run_500_pred,
  `ranknet+aux(lr=0.01)` = run_pred_lr01,
  `ranknet+aux(lr=0.001)` = run_pred_lr001,
  .id = "model"
) %>%
  group_by(model) %>%
  mutate(
    across(c(value, prediction), scale)
  ) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(model)) +
  theme(aspect.ratio = 1)

bind_rows(
  `ranknet` = run_500_pred,
  `ranknet+aux` = run_pred,
  .id = "model"
) %>%
  group_by(model) %>%
  mutate(
    across(c(value, prediction), \(x) (rank(x) - 1) / (length(x) - 1))
  ) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5) +
  facet_wrap(vars(model), ncol = 1)


# aux2 --------------------------------------------------------------------

run_name <- "03_ranknet+scalar_500_encoder_aux2"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)

# use pairs and images from ranknet(200)
# but add date column for aux lookup
file.copy(
  file.path("runs", "02_ranknet+scalar_500", "stations", station_id, "input", "pairs.csv"),
  file.path(run_station_dir, "input", "pairs.csv"),
  overwrite = TRUE
)
file.copy(
  file.path("runs", "02_ranknet+scalar_500", "stations", station_id, "input", "images.csv"),
  file.path(run_station_dir, "input", "images.csv"),
  overwrite = TRUE
)
aux_day %>%
  write_csv(file.path(run_station_dir, "input", "aux.csv"))


run_name <- "03_ranknet+scalar_500_encoder_aux2_drop2"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)

file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "pairs.csv"),
  file.path(run_station_dir, "input", "pairs.csv"),
  overwrite = TRUE
)
file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "images.csv"),
  file.path(run_station_dir, "input", "images.csv"),
  overwrite = TRUE
)
file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "aux.csv"),
  file.path(run_station_dir, "input", "aux.csv"),
  overwrite = TRUE
)
list(
  aux_encoder_layers = "[64, 32]",
  aux_encoder_dropout = 0.2
) %>%
  write_yaml(file.path(run_station_dir, "input", "config.yml"))


# remove aux_encoder_dropout, set aux_encoder_layers to [128, 64]
run_name <- "03_ranknet+scalar_500_encoder_aux2_128_64"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)
file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "pairs.csv"),
  file.path(run_station_dir, "input", "pairs.csv"),
  overwrite = TRUE
)
file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "images.csv"),
  file.path(run_station_dir, "input", "images.csv"),
  overwrite = TRUE
)
file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "aux.csv"),
  file.path(run_station_dir, "input", "aux.csv"),
  overwrite = TRUE
)
list(
  aux_encoder_layers = "[128, 64]",
  aux_encoder_dropout = 0.0
) %>%
  write_yaml(file.path(run_station_dir, "input", "config.yml"))



pred_concat_none <- read_csv(file.path("runs", "02_ranknet+scalar_500_lr01_no_aux", "stations", station_id, "output", "data", "predictions.csv"))
pred_concat_aux1 <- read_csv(file.path("runs", "02_ranknet+scalar_500_lr01", "stations", station_id, "output", "data", "predictions.csv"))
pred_encoder_aux1 <- read_csv(file.path("runs", "03_ranknet+scalar_500_encoder", "stations", station_id, "output", "data", "predictions.csv"))
pred_encoder_aux2 <- read_csv(file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "output", "data", "predictions.csv"))
pred_encoder_aux2_drop2 <- read_csv(file.path("runs", "03_ranknet+scalar_500_encoder_aux2_drop2", "stations", station_id, "output", "data", "predictions.csv"))
pred_encoder_aux2_128_64 <- read_csv(file.path("runs", "03_ranknet+scalar_500_encoder_aux2_128_64", "stations", station_id, "output", "data", "predictions.csv"))
# aux2 helps with high flow estimates
# dropout rate doesn't matter much
# encoder is better than concat
# aux2 encoding layers: 64,32 better than 128,64


bind_rows(
  none = pred_concat_none,
  concat = pred_concat_aux1,
  aux1 = pred_encoder_aux1,
  aux2 = pred_encoder_aux2,
  aux2_drop2 = pred_encoder_aux2_drop2,
  aux2_128_64 = pred_encoder_aux2_128_64,
  .id = "run"
) %>%
  mutate(run = fct_inorder(run)) %>%
  group_by(run, split) %>%
  summarise(
    # n = n(),
    tau = cor(prediction, value, method = "kendall")
  ) %>%
  pivot_wider(names_from = "run", values_from = "tau")

bind_rows(
  none = pred_concat_none,
  concat = pred_concat_aux1,
  aux1 = pred_encoder_aux1,
  aux2 = pred_encoder_aux2,
  aux2_drop2 = pred_encoder_aux2_drop2,
  aux2_128_64 = pred_encoder_aux2_128_64,
  .id = "run"
) %>%
  mutate(run = fct_inorder(run)) %>%
  group_by(run) %>%
  mutate(
    across(c(value, prediction), scale)
  ) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(run)) +
  theme(aspect.ratio = 1)

bind_rows(
  none = pred_concat_none,
  concat = pred_concat_aux1,
  aux1 = pred_encoder_aux1,
  aux2 = pred_encoder_aux2,
  aux2_drop2 = pred_encoder_aux2_drop2,
  aux2_128_64 = pred_encoder_aux2_128_64,
  .id = "run"
) %>%
  mutate(run = fct_inorder(run)) %>%
  group_by(run) %>%
  mutate(
    across(c(value, prediction), scale)
  ) %>%
  filter(year(date) == 2023, month(date) %in% 6:8) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5) +
  facet_wrap(vars(run), ncol = 1)

bind_rows(
  none = pred_concat_none,
  concat = pred_concat_aux1,
  aux1 = pred_encoder_aux1,
  aux2 = pred_encoder_aux2,
  aux2_drop2 = pred_encoder_aux2_drop2,
  aux2_128_64 = pred_encoder_aux2_128_64,
  .id = "run"
) %>%
  group_by(run) %>%
  mutate(
    across(c(value, prediction), \(x) (rank(x) - 1) / (length(x) - 1))
  ) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5) +
  facet_wrap(vars(run), ncol = 1)


# lstm --------------------------------------------------------------------

run_name <- "04_ranknet+scalar_500_lstm"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)

file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "pairs.csv"),
  file.path(run_station_dir, "input", "pairs.csv"),
  overwrite = TRUE
)
file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "images.csv"),
  file.path(run_station_dir, "input", "images.csv"),
  overwrite = TRUE
)
# using rolling stat aux variables from encoder
file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "aux.csv"),
  file.path(run_station_dir, "input", "aux.csv"),
  overwrite = TRUE
)

run_name <- "04_ranknet+scalar_500_lstm_ts"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)

file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "pairs.csv"),
  file.path(run_station_dir, "input", "pairs.csv"),
  overwrite = TRUE
)
file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "images.csv"),
  file.path(run_station_dir, "input", "images.csv"),
  overwrite = TRUE
)
# using daily aux data only
aux_day_inp %>%
  write_csv(file.path(run_station_dir, "input", "aux.csv"))

# re-run concat with current aux
run_name <- "04_ranknet+scalar_500_concat"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)

file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "pairs.csv"),
  file.path(run_station_dir, "input", "pairs.csv"),
  overwrite = TRUE
)
file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "images.csv"),
  file.path(run_station_dir, "input", "images.csv"),
  overwrite = TRUE
)
aux_day %>%
  write_csv(file.path(run_station_dir, "input", "aux.csv"))

# re-run concat with current aux
run_name <- "04_ranknet+scalar_500_none"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)

file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "pairs.csv"),
  file.path(run_station_dir, "input", "pairs.csv"),
  overwrite = TRUE
)
file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "images.csv"),
  file.path(run_station_dir, "input", "images.csv"),
  overwrite = TRUE
)
# aux_day %>%
#   write_csv(file.path(run_station_dir, "input", "aux.csv"))

# re-run encoder
run_name <- "04_ranknet+scalar_500_encoder"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)

file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "pairs.csv"),
  file.path(run_station_dir, "input", "pairs.csv"),
  overwrite = TRUE
)
file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "images.csv"),
  file.path(run_station_dir, "input", "images.csv"),
  overwrite = TRUE
)
aux_day %>%
  write_csv(file.path(run_station_dir, "input", "aux.csv"))

# met: hr -----------------------------------------------------------------

openmeteo::weather_variables()$hourly_history_vars

met_hr <- openmeteo::weather_history(
  location = c(station$latitude, station$longitude),
  start = as.character(met_start),
  end = as.character(met_end),
  timezone = "UTC",
  hourly = c(
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "vapour_pressure_deficit",
    "rain",
    "snowfall",
    "soil_moisture_0_to_7cm"
  )
)

met_hr %>%
  pivot_longer(-datetime) %>%
  ggplot(aes(datetime, value)) +
  geom_line() +
  facet_wrap(vars(name), scales = "free_y")

aux_hr <- met_hr %>%
  transmute(
    timestamp = datetime,
    airtemp = hourly_temperature_2m,
    relhum = hourly_relative_humidity_2m,
    pressure = hourly_pressure_msl,
    wspeed = hourly_wind_speed_10m,
    vpd = hourly_vapour_pressure_deficit,
    rain = hourly_rain,
    snowfalL = hourly_snowfall,
    soilmoisture = hourly_soil_moisture_0_to_7cm
  ) %>%
  mutate(
    across(-timestamp, \(x) (x - min(x)) / (max(x) - min(x)))
  ) %>%
  print()



# lstm hr -----------------------------------------------------------------


run_name <- "04_ranknet+scalar_500_lstm_H"
run_station_dir <- file.path("runs", run_name, "stations", station_id)
dir.create(file.path(run_station_dir, "input"), showWarnings = FALSE, recursive = TRUE)

file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "pairs.csv"),
  file.path(run_station_dir, "input", "pairs.csv"),
  overwrite = TRUE
)
file.copy(
  file.path("runs", "03_ranknet+scalar_500_encoder_aux2", "stations", station_id, "input", "images.csv"),
  file.path(run_station_dir, "input", "images.csv"),
  overwrite = TRUE
)

aux_hr %>%
  write_csv(file.path(run_station_dir, "input", "aux.csv"))




# compare -----------------------------------------------------------------

pred_none <- read_csv(file.path("runs", "04_ranknet+scalar_500_none", "stations", station_id, "output", "data", "predictions.csv"))
pred_concat <- read_csv(file.path("runs", "04_ranknet+scalar_500_concat", "stations", station_id, "output", "data", "predictions.csv"))
pred_encoder <- read_csv(file.path("runs", "04_ranknet+scalar_500_encoder", "stations", station_id, "output", "data", "predictions.csv"))
pred_lstm <- read_csv(file.path("runs", "04_ranknet+scalar_500_lstm", "stations", station_id, "output", "data", "predictions.csv"))
pred_lstm_ts <- read_csv(file.path("runs", "04_ranknet+scalar_500_lstm_ts", "stations", station_id, "output", "data", "predictions.csv"))
pred_lstm_hr <- read_csv(file.path("runs", "04_ranknet+scalar_500_lstm_H", "stations", station_id, "output", "data", "predictions.csv"))

bind_rows(
  none = pred_none,
  concat = pred_concat,
  encoder = pred_encoder,
  lstm = pred_lstm,
  lstm_ts = pred_lstm_ts,
  lstm_hr = pred_lstm_hr,
  .id = "run"
) %>%
  mutate(run = fct_inorder(run)) %>%
  group_by(run, split) %>%
  summarise(
    # n = n(),
    tau = cor(prediction, value, method = "kendall")
  ) %>%
  pivot_wider(names_from = "run", values_from = "tau")

bind_rows(
  none = pred_none,
  concat = pred_concat,
  encoder = pred_encoder,
  lstm = pred_lstm,
  lstm_ts = pred_lstm_ts,
  lstm_hr = pred_lstm_hr,
  .id = "run"
) %>%
  mutate(run = fct_inorder(run)) %>%
  group_by(run) %>%
  mutate(
    across(c(value, prediction), scale)
  ) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(run)) +
  theme(aspect.ratio = 1)

bind_rows(
  none = pred_none,
  concat = pred_concat,
  encoder = pred_encoder,
  lstm = pred_lstm,
  lstm_ts = pred_lstm_ts,
  lstm_hr = pred_lstm_hr,
  .id = "run"
) %>%
  mutate(run = fct_inorder(run)) %>%
  group_by(run) %>%
  mutate(
    across(c(value, prediction), \(x) (rank(x) - 1) / (length(x) - 1))
  ) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(run)) +
  theme(aspect.ratio = 1)

bind_rows(
  none = pred_none,
  concat = pred_concat,
  encoder = pred_encoder,
  lstm = pred_lstm,
  lstm_ts = pred_lstm_ts,
  lstm_hr = pred_lstm_hr,
  .id = "run"
) %>%
  mutate(run = fct_inorder(run)) %>%
  group_by(run) %>%
  mutate(
    across(c(value, prediction), scale)
  ) %>%
  filter(year(date) == 2023, month(date) %in% 6:8) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5) +
  facet_wrap(vars(run), ncol = 1)

bind_rows(
  none = pred_none,
  concat = pred_concat,
  encoder = pred_encoder,
  lstm = pred_lstm,
  lstm_ts = pred_lstm_ts,
  lstm_hr = pred_lstm_hr,
  .id = "run"
) %>%
  group_by(run) %>%
  mutate(
    across(c(value, prediction), \(x) (rank(x) - 1) / (length(x) - 1))
  ) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5) +
  facet_wrap(vars(run), ncol = 1)

setwd("~/git/fpe-model/experiments/20250120-aux")

library(tidyverse)
library(janitor)
library(glue)
library(daymetr)
library(openmeteo)

station_id <- 29

hourly_vars <- openmeteo::weather_variables()$hourly_history_vars


station <- read_json(file.path("stations", station_id, "data", "station.json"))
station_images <- read_csv(file.path("stations", station_id, "data", "images.csv")) %>%
  mutate(
    timestamp_hour = hour(with_tz(timestamp, tzone = station$timezone)),
    value = log10(pmax(value, 0.01))
  ) %>%
  filter(
    !is.na(value),
    timestamp_hour %in% 8:16
  )

test_start <- ymd(20240101)

tabyl(station_images, timestamp_hour)
station_pairs <- read_csv(file.path("stations", station_id, "model", "pairs.csv")) %>%
  filter(
    image_id_1 %in% station_images$image_id,
    image_id_2 %in% station_images$image_id,
    as_date(with_tz(timestamp_1, tzone = station$timezone)) < test_start,
    as_date(with_tz(timestamp_2, tzone = station$timezone)) < test_start
  ) %>%
  mutate(
    across(starts_with("value_"), ~ log10(pmax(., 0.01)))
  )
summary(station_pairs)

start <- as_date(min(images$timestamp)) - days(30)
end <- as_date(max(images$timestamp))

met <- openmeteo::weather_history(
  location = c(station$latitude, station$longitude),
  start = as.character(start),
  end = as.character(end),
  timezone = "UTC",
  hourly = c(
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "vapour_pressure_deficit",
    "precipitation",
    "rain",
    "snowfall",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm"
  )
)

met %>%
  pivot_longer(-datetime) %>%
  ggplot(aes(datetime, value)) +
  geom_line() +
  facet_wrap(vars(name), scales = "free_y")

# met %>%
#   ggplot(aes(hourly_dew_point_2m, hourly_temperature_2m, color = hourly_relative_humidity_2m)) +
#   geom_point(alpha = 0.5, size = 0.5) +
#   scale_color_viridis_c()


# daymet ------------------------------------------------------------------

daymet_raw <- daymetr::download_daymet(
  lat = station$latitude,
  lon = station$longitude,
  start = year(start),
  end = pmin(year(end), 2023),
)

daymet <- daymet_raw$data %>%
  clean_names() %>%
  as_tibble() %>%
  mutate(
    date = ymd(glue("{year}-01-01")) + days(yday - 1)
  ) %>%
  complete(date = seq.Date(from = min(date), to = max(date), by = "day")) %>%
  transmute(
    date,
    dayl = dayl_s,
    prcp = prcp_mm_day,
    srad = srad_w_m_2,
    swe = swe_kg_m_2,
    tmax = tmax_deg_c,
    tmin = tmin_deg_c,
    vp = vp_pa
  )


# compare met to daymet ---------------------------------------------------

met_day <- met %>%
  mutate(
    datetime_local = with_tz(datetime, tzone = "US/Eastern")
  ) %>%
  group_by(date = as_date(datetime_local)) %>%
  summarize(
    prcp = sum(hourly_precipitation),
    tmax = max(hourly_temperature_2m),
    tmin = min(hourly_temperature_2m)
  ) %>%
  filter(date <= max(daymet$date))

bind_rows(
  met = met_day,
  daymet = daymet,
  .id = "source"
) %>%
  select(source, date, prcp, tmax, tmin) %>%
  pivot_longer(-c(source, date)) %>%
  ggplot(aes(date, value)) +
  geom_point(aes(color = source)) +
  facet_wrap(vars(name), scales = "free_y")

bind_rows(
  met = met_day,
  daymet = daymet,
  .id = "source"
) %>%
  select(source, date, prcp, tmax, tmin) %>%
  pivot_longer(-c(source, date)) %>%
  pivot_wider(names_from = "source") %>%
  ggplot(aes(met, daymet)) +
  geom_abline() +
  geom_point(aes()) +
  geom_blank(aes(daymet, met)) +
  facet_wrap(vars(name), scales = "free_y") +
  theme(aspect.ratio = 1)

bind_rows(
  met = met_day,
  daymet = daymet,
  .id = "source"
) %>%
  filter(year(date) == 2022, month(date) == 6) %>%
  select(source, date, prcp, tmax, tmin) %>%
  ggplot(aes(date, prcp)) +
  geom_point(aes(color = source))

bind_rows(
  met = met_day,
  daymet = daymet,
  .id = "source"
) %>%
  group_by(source, date = floor_date(date, "month")) %>%
  summarise(prcp = sum(prcp)) %>%
  pivot_wider(names_from = "source", values_from = "prcp") %>%
  ggplot(aes(met, daymet)) +
  geom_abline() +
  geom_point(aes()) +
  geom_blank(aes(daymet, met)) +
  theme(aspect.ratio = 1)

# aux
# normalize met to 0-1
aux <- met %>%
  rename(timestamp = datetime) %>%
  mutate(
    across(-timestamp, \(x) (x - min(x)) / (max(x) - min(x)))
  )
summary(aux)

aux %>%
  pivot_longer(-datetime) %>%
  ggplot(aes(datetime, value)) +
  geom_line() +
  facet_wrap(vars(name))

# training dataset
# - pairs(100, 250, 500)
# - images(200, 500, 1000, 5000)

# frac_train <- 0.8

set.seed(1105)
n_val_pairs <- 100
run_pairs_val <- station_pairs %>%
  filter(split == "val") %>%
  nest_by(pair) %>%
  ungroup() %>%
  slice_sample(n = n_val_pairs) %>%
  unnest(data)

set.seed(1113)
run_inp <- tibble(
  n_train_pairs = c(100, 200, 500),
) %>%
  crossing(trial = 1:5) %>%
  rowwise() %>%
  mutate(
    dataset = glue("pairs_{n_train_pairs}-trial_{trial}"),
    pairs = list({
      x_train <- station_pairs %>%
        filter(split == "train") %>%
        nest_by(pair) %>%
        ungroup() %>%
        slice_sample(n = n_train_pairs) %>%
        unnest(data)
      bind_rows(x_train, run_pairs_val)
    }),
    images = list({
      x_train_pairs <- pairs %>%
        filter(split == "train")
      x_val_pairs <- pairs %>%
        filter(split == "val")
      station_images %>%
        mutate(
          split = case_when(
            image_id %in% c(x_train_pairs$image_id_1, x_train_pairs$image_id_2) ~ "train",
            image_id %in% c(x_val_pairs$image_id_1, x_val_pairs$image_id_2) ~ "val",
            as_date(with_tz(timestamp, tzone = station$timezone)) < test_start ~ "test-in",
            TRUE ~ "test-out"
          )
        )
    })
  ) %>%
  ungroup() %>%
  print()

run_inp %>%
  unnest(pairs) %>%
  ggplot(aes(timestamp_1, timestamp_2, color = split)) +
  geom_point() +
  facet_grid(vars(n_train_pairs), vars(trial))

run_inp %>%
  unnest(pairs) %>%
  ggplot(aes(value_1, value_2, color = factor(label))) +
  geom_point() +
  geom_blank(aes(value_2, value_1)) +
  facet_grid(vars(n_train_pairs), vars(trial)) +
  theme(aspect.ratio = 1)

run_inp %>%
  unnest(images) %>%
  ggplot(aes(timestamp, value)) +
  geom_line() +
  geom_point(aes(color = split), size = 0.5) +
  facet_grid(vars(n_train_pairs), vars(trial))

run_inp %>%
  unnest(images) %>%
  filter(split == "train") %>%
  ggplot(aes(factor(trial), value)) +
  geom_boxplot() +
  facet_grid(vars(), vars(n_train_pairs))



# generate input files ----------------------------------------------------


runs <- tibble(
  station_id = station$id,
  model = c("resnet", "lstm", "resnet+lstm", "ranknet", "ranknet+lstm")
) %>%
  crossing(run_inp) %>%
  rowwise() %>%
  mutate(
    run_name = glue("{model}-s_{station_id}-n_{n_train_pairs}-t_{trial}"),
    run_dir = {
      run_dir <- file.path("runs", run_name)
      input_dir <- file.path(run_dir, "input")
      dir.create(input_dir, showWarnings = FALSE, recursive = TRUE)
      dir.create(file.path(run_dir, "output"), showWarnings = FALSE, recursive = TRUE)
      run_dir
    },
    aux_filename = {
      f <- file.path(run_dir, "input", "aux.csv")
      aux %>%
        write_csv(f)
      f
    },
    pairs_filename = {
      f <- file.path(run_dir, "input", "pairs.csv")
      pairs %>%
        write_csv(f)
      f
    },
    images_filename = {
      f <- file.path(run_dir, "input", "images.csv")
      images %>%
        write_csv(f)
      f
    }
  ) %>%
  print()

runs$run_name %>%
  write_lines("runs/runs.txt")



# lstm-test ---------------------------------------------------------------

run_name <- "dev-lstm"
dir.create(file.path("runs", run_name, "input"), showWarnings = FALSE, recursive = TRUE)
runs$pairs[[1]] %>%
  write_csv(file.path(file.path("runs", run_name, "input", "pairs.csv")))
set.seed(1626)
runs$images[[1]] %>%
  mutate(
    r = runif(n = n()),
    split = case_when(
      as_date(with_tz(timestamp, tzone = station$timezone)) >= test_start ~ "test-out",
      # r > 0.5 ~ "test-in",
      r > 0.8 ~ "val",
      TRUE ~ "train"
    )
  ) %>%
  # filter(split != "test-in") %>%
  nest_by(split) %>%
  mutate(
    data = list({
      if (split == "test-out") {
        slice_sample(data, n = 1000)
      } else {
        data
      }
    })
  ) %>%
  ungroup() %>%
  unnest(data) %>%
  write_csv(file.path(file.path("runs", run_name, "input", "images.csv")))

aux %>%
  write_csv(file.path(file.path("runs", run_name, "input", "aux.csv")))

dev_lstm <- read_csv(file.path("runs", run_name, "output", "data", "predictions.csv"))

dev_lstm %>%
  group_by(split) %>%
  summarise(
    rmse = sqrt(mean((value - prediction) ^ 2)),
    tau = cor(value, prediction, method = "kendall")
  )
dev_lstm %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split)) +
  geom_blank(aes(prediction, value))

dev_lstm %>%
  ggplot(aes(timestamp)) +
  geom_point(aes(y = value, color = split)) +
  geom_line(aes(y = prediction), color = "red")


# dev-lstm-day ------------------------------------------------------------

run_name <- "dev-lstm-day"
dir.create(file.path("runs", run_name, "input"), showWarnings = FALSE, recursive = TRUE)
set.seed(1626)
values <- read_csv(glue("stations/{station$id}/data/values.csv")) %>%
  filter(!is.na(value))
daily_values <- values %>%
  group_by(date = as_date(with_tz(timestamp, tz = station$timezone))) %>%
  summarise(
    n = n(),
    value = log10(mean(value))
  ) %>%
  filter(
    n >= 90,
    date >= min(as_date(met$datetime)) + days(90),
    date <= max(as_date(met$datetime))
  )
daily_values %>%
  ggplot(aes(date, value)) +
  geom_line()

daily_values %>%
  filter(date < ymd(20240101)) %>%
  mutate(
    r = runif(n = n()),
    split = case_when(
      date >= test_start - months(6) ~ "test-out",
      date >= test_start - months(12) ~ "val",
      # r > 0.8 ~ "val",
      # r > 0.4 ~ "val",
      TRUE ~ "train"
    )
  ) %>%
  # filter(split != "test-in") %>%
  nest_by(split) %>%
  mutate(
    # data = list({
    #   if (split == "test-out") {
    #     slice_sample(data, n = 1000)
    #   } else {
    #     data
    #   }
    # })
  ) %>%
  ungroup() %>%
  unnest(data) %>%
  write_csv(file.path(file.path("runs", run_name, "input", "images.csv")))

met %>%
  mutate(
    date = as_date(with_tz(datetime, tzone = "US/Eastern"))
  ) %>%
  group_by(date) %>%
  summarize(
    across(-c(datetime), mean)
  ) %>%
  mutate(
    across(-date, \(x) (x - min(x)) / (max(x) - min(x)))
  ) %>%
  filter(year(date) < 2024) %>%
  write_csv(file.path(file.path("runs", run_name, "input", "aux.csv")))

daymet %>%
  mutate(
    across(-date, \(x) (x - min(x)) / (max(x) - min(x)))
  ) %>%
  write_csv(file.path(file.path("runs", run_name, "input", "aux.csv")))


dev_lstm_day_daymet <- read_csv(file.path("runs", run_name, "output", "data", "predictions.csv"))

bind_rows(
  # met = dev_lstm_day,
  daymet = dev_lstm_day_daymet,
  .id = "source"
) %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split)) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(source)) +
  theme(aspect.ratio = 1)

dev_lstm_day %>%
  group_by(split) %>%
  summarise(
    rmse = sqrt(mean((value - prediction) ^ 2)),
    tau = cor(value, prediction, method = "kendall")
  )
dev_lstm_day_daymet %>%
  ggplot(aes(date)) +
  geom_line(aes(y = value), color = "black") +
  geom_point(aes(y = prediction, color = split))



# long-term dataset -------------------------------------------------------




# images ------------------------------------------------------------------

images <- read_csv("data/images.csv") %>%
  mutate(
    timestamp = with_tz(timestamp, "America/New_York"),
    date = as_date(timestamp)
  ) %>%
  filter(
    year(timestamp) <= 2023,
    !is.na(value)
  ) %>%
  mutate(
    value = log10(value)
  ) %>%
  select(-split)

images_train_val <- images %>%
  filter(date < ymd(20230601))
images_train <- images_train_val %>%
  sample_n(size = 1000) %>%
  mutate(split = "train") %>%
  select(split, image_id, filename, timestamp, date, value) %>%
  print()
images_val <- images_train_val %>%
  filter(!image_id %in% images_train$image_id) %>%
  sample_n(size = 200) %>%
  mutate(split = "val") %>%
  select(split, image_id, filename, timestamp, date, value) %>%
  print()
images_test_in <- images_train_val %>%
  filter(
    !image_id %in% images_train$image_id,
    !image_id %in% images_val$image_id
  ) %>%
  mutate(split = "test-in") %>%
  select(split, image_id, filename, timestamp, date, value)
images_test_out <- images %>%
  filter(date > max(images_train_val$date)) %>%
  mutate(split = "test-out") %>%
  select(split, image_id, filename, timestamp, date, value)

images_split <- bind_rows(
  images_train,
  images_val,
  images_test_in,
  images_test_out
)

images_split %>%
  ggplot(aes(timestamp, value)) +
  geom_point(aes(color = split), size = 0.5)

tabyl(images_split, split)

images_split %>%
  mutate(
    across(starts_with("timestamp"), ~ format(with_tz(., "UTC"), "%Y-%m-%d %H:%M:%S"))
  ) %>%
  write_csv("data/images-split.csv")

# aux: daymet ------------------------------------------------------------------

aux_daymet_raw <- daymetr::download_daymet(
  site = "AVERYBB",
  lat = coords[[1]],
  lon = coords[[2]],
  start = min(year(images$timestamp)),
  end = 2023,
)

aux_daymet <- aux_daymet_raw$data %>%
  clean_names() %>%
  as_tibble() %>%
  mutate(
    date = ymd(glue("{year}-01-01")) + days(yday - 1)
  ) %>%
  complete(date = seq.Date(from = min(date), to = max(date), by = "day")) %>%
  transmute(
    date,
    dayl = dayl_s,
    prcp = prcp_mm_day,
    srad = srad_w_m_2,
    swe = swe_kg_m_2,
    tmax = tmax_deg_c,
    tmin = tmin_deg_c,
    vp = vp_pa
  ) %>%
  mutate(
    across(
      -date,
      \(x) scale(x)[,1], .names = "{.col}_z"
    )
  ) %>%
  print()


aux_daymet %>%
  write_csv("data/aux.csv")


# run 01-resnet ------------------------------------------------------------------

dir.create("runs/01-resnet/data", showWarnings = FALSE, recursive = TRUE)
file.copy("data/images-split.csv", "runs/01-resnet/data/images.csv", overwrite = TRUE)


x_1 <- read_csv("runs/01-resnet/output/data/predictions.csv") %>%
  arrange(timestamp)


# run 02-lstm -------------------------------------------------------------

dir.create("runs/02-lstm/data", showWarnings = FALSE, recursive = TRUE)
file.copy("data/images-split.csv", "runs/02-lstm/data/labels.csv", overwrite = TRUE)
file.copy("data/aux.csv", "runs/02-lstm/data/aux.csv", overwrite = TRUE)

x_2 <- read_csv("runs/02-lstm/output/data/02-lstm-predictions.csv") %>%
  arrange(timestamp)

x_2 %>%
  ggplot(aes(value, prediction)) +
  geom_point(size = 0.1)

x_2 %>%
  ggplot(aes(rank(value), rank(prediction))) +
  geom_point(size = 0.1)

x_2 %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value, color = "obs")) +
  geom_line(aes(y = prediction, color = "pred"))

bind_rows(
  resnet = x_1,
  lstm = x_2,
  .id = "model"
) %>%
  ggplot(aes(value, prediction, color = split)) +
  geom_point(size = 0.1) +
  facet_wrap(vars(model))


# tau vs n images ---------------------------------------------------------

x <- read_csv("~/data/fpe/experiments/20241111-WB0-n_train/runs/n_train_1000/output/test/predictions.csv")

x_tau <- tibble(
  n = c(seq(10, 90, by = 10), seq(100, 16000, by = 100))
) %>%
  mutate(
    data = map(n, \(n) sample_n(x, size = n)),
    tau = map_dbl(data, \(x) cor(x$value, x$score, method = "kendall"))
  )
x_tau %>%
  ggplot(aes(n, tau)) +
  geom_point()


# run 03-resnet-lstm ---------------------------------------------------------


dir.create("runs/03-resnet-lstm/data", showWarnings = FALSE, recursive = TRUE)
file.copy("data/images-split.csv", "runs/03-resnet-lstm/data/images.csv", overwrite = TRUE)
file.copy("data/aux.csv", "runs/03-resnet-lstm/data/aux.csv", overwrite = TRUE)


x_3 <- read_csv("runs/03-resnet-lstm/output/data/03-resnet-lstm-predictions.csv") %>%
  arrange(timestamp)


bind_rows(
  resnet = x_1,
  lstm = x_2,
  .id = "model"
) %>%
  group_by(model, date = as_date(timestamp)) %>%
  summarise(across(c(value, prediction), mean)) %>%
  ggplot(aes(value, prediction)) +
  geom_point(size = 0.1) +
  facet_wrap(vars(model))




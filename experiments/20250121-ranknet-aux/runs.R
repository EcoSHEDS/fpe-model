setwd("~/git/fpe-model/experiments/20250121-ranknet-aux/")

library(tidyverse)
library(janitor)
library(glue)
library(daymetr)
library(openmeteo)

experiment <- "20250121-ranknet-aux"
daytime_hours <- 8:16

start_test <- ymd(20240101)

# stations ----------------------------------------------------------------

load_station <- function (station_id) {
  station <- read_json(file.path("stations", station_id, "data", "station.json"))
  images <- read_csv(file.path("stations", station_id, "data", "images.csv"), show_col_types = FALSE) %>%
    mutate(
      date = as_date(with_tz(timestamp, tzone = station$timezone)),
      timestamp_hour = hour(with_tz(timestamp, tzone = station$timezone)),
    ) %>%
    filter(
      !is.na(value),
      value > 0,
      timestamp_hour %in% daytime_hours
    ) %>%
    mutate(
      value = log10(value),
      file_downloaded = file.exists(file.path("~/data/fpe/images", filename)),
    )

  f <- file.path("stations", station_id, "model", "pairs.csv")
  if (file.exists(f)) {
    pairs <- read_csv(f, show_col_types = FALSE) %>%
      mutate(
        across(starts_with("value_"), ~ log10(.)),
        date_1 = as_date(with_tz(timestamp_1, tzone = station$timezone)),
        date_2 = as_date(with_tz(timestamp_2, tzone = station$timezone)),
        timestamp_hour_1 = hour(with_tz(timestamp_1, tzone = station$timezone)),
        timestamp_hour_2 = hour(with_tz(timestamp_2, tzone = station$timezone)),
        file_downloaded_1 = file.exists(file.path("~/data/fpe/images", filename_1)),
        file_downloaded_2 = file.exists(file.path("~/data/fpe/images", filename_2)),
      ) %>%
      filter(
        image_id_1 %in% images$image_id,
        image_id_2 %in% images$image_id,
        timestamp_hour_1 %in% daytime_hours,
        timestamp_hour_2 %in% daytime_hours
      )
  } else {
    pairs <- tibble()
  }

  list(
    station = station,
    images = images,
    pairs = pairs
  )
}

stations_all <- tibble(
  station_id = as.numeric(basename(list.dirs("stations", recursive = FALSE)))
) %>%
  mutate(
    data = map(station_id, load_station)
  ) %>%
  unnest_wider(data) %>%
  mutate(has_pairs = map_lgl(pairs, ~ !is.null(.))) %>%
  filter(has_pairs) %>%
  select(-has_pairs) %>%
  rowwise() %>%
  mutate(
    station_code = glue("{station_id}-{station$name}"),
    images_n = nrow(images),
    images_n_downloaded = sum(images$file_downloaded),
    images_start = min(images$timestamp),
    images_end = max(images$timestamp),
    images_duration_days = as.numeric(difftime(images_end, images_start, "days")),
    test_start = {
      image_dates <- unique(images$date)
      floor_date(as.Date(quantile(as.numeric(image_dates), probs = 0.8)), "month")
    },
    pairs = list({
      pairs %>%
        filter(date_1 < test_start, date_2 < test_start)
    }),
    pairs_n = nrow(pairs) / 2,
    pairs_n_downloaded = sum(pairs$file_downloaded_1 & pairs$file_downloaded_2) / 2,
    pairs_start = min(c(pairs$timestamp_1, pairs$timestamp_2)),
    pairs_end = max(c(pairs$timestamp_1, pairs$timestamp_2)),
    .before = "station_id"
  ) %>%
  print()

stations <- stations_all %>%
  filter(images_duration_days > 700, pairs_n_downloaded > 600)

stations %>%
  ggplot() +
  geom_segment(
    aes(as_date(images_start), xend = as_date(images_end), y = station_code, yend = station_code)
  ) +
  geom_segment(
    aes(test_start, xend = as_date(images_end), y = station_code, yend = station_code),
    color = "red"
  )

stations %>%
  select(station_code, test_start, images) %>%
  unnest(images) %>%
  mutate(split = if_else(timestamp >= test_start, "test", "train")) %>%
  ggplot(aes(timestamp, value)) +
  geom_line(aes(color = split)) +
  facet_wrap(vars(station_code), scales = "free")

stations %>%
  transmute(id = station$id) %>%
  pull(id) %>%
  sort() %>%
  write_lines("stations/stations.txt")


# met data ----------------------------------------------------------------

.query_openmeteo <- function(
    location,
    start,
    end,
    hourly = NULL,
    daily = NULL,
    response_units = NULL,
    model = NULL,
    timezone = "auto",
    downscaling = NULL,
    base_url = "https://customer-archive-api.open-meteo.com/v1/archive",
    apikey = Sys.getenv("OPENMETEO_API_KEY")) {
  coordinates <- openmeteo:::.coords_generic(location)

  # base queries
  queries <- list(
    latitude = coordinates[1],
    longitude = coordinates[2],
    start_date = start,
    end_date = end,
    timezone = timezone,
    apikey = apikey
  )

  # add units/hourly/daily/model as supplied
  queries <- c(queries, response_units)
  if (!is.null(hourly)) {
    queries$hourly <- paste(hourly, collapse = ",")
  }
  if (!is.null(daily)) {
    queries$daily <- paste(daily, collapse = ",")
  }
  if (!is.null(model)) {
    if (length(model) != 1) {
      stop("Please specify only one model per query.") # may support later
    }
    queries$models <- paste(model, collapse = ",")
  }

  ## handle downscaling switch for climate forecast
  if(!is.null(downscaling))queries[["disable_bias_correction"]] <- paste(!downscaling, collapse = ",")

  # request (decode necessary as API treats ',' differently to '%2C')
  pl <- httr::GET(utils::URLdecode(httr::modify_url(base_url, query = queries)))
  openmeteo:::.response_OK(pl)
  Sys.sleep(5)

  # parse response
  pl_parsed <- httr::content(pl, as = "parsed")

  tz <- pl_parsed$timezone
  dtformat <- "%Y-%m-%dT%H:%M"
  export_both <- (!is.null(hourly) & !is.null(daily))

  # parse hourly data
  if (!is.null(pl_parsed$hourly)) {
    hourly_tibble <-
      pl_parsed$hourly |>
      openmeteo:::.nestedlist_as_tibble() |>
      dplyr::rename_with(~ paste0("hourly_", .x), .cols = -time) |>
      dplyr::mutate(datetime = as.POSIXct(time, format = dtformat, tz = tz)) |>
      dplyr::relocate(datetime, .before = time) |>
      dplyr::select(-time)

    if (!export_both) {
      return(hourly_tibble)
    }
  }

  # process daily data
  if (!is.null(pl_parsed$daily)) {
    daily_tibble <-
      pl_parsed$daily |>
      openmeteo:::.nestedlist_as_tibble() |>
      dplyr::rename_with(~ paste0("daily_", .x), .cols = -time) |>
      dplyr::mutate(date = as.Date(time, tz = tz)) |>
      dplyr::relocate(date, .before = time) |>
      dplyr::select(-time)

    if (!export_both) {
      return(daily_tibble)
    }
  }

  # combine both hourly and daily if requested
  if (export_both) {
    d <-
      daily_tibble |>
      dplyr::mutate(date = as.character(date))

    h <-
      hourly_tibble |>
      dplyr::mutate(date = as.character(datetime)) |>
      dplyr::select(-datetime)

    dh <-
      dplyr::full_join(d, h, by = "date") |>
      tidyr::separate(
        col = "date",
        sep = " ",
        fill = "right",
        into = c("date", "time")
      ) |>
      dplyr::mutate(date = as.Date(date, tz = tz))

    return(dh)
  }
}

get_met <- function (location, start, end, hourly = NULL, daily = NULL, timezone = "auto") {
  .query_openmeteo(
    location = location,
    start = start,
    end = end,
    hourly = hourly,
    daily = daily,
    timezone = timezone
  )
}

# get_met(
#   latitude = stations$station[[1]]$latitude,
#   longitude = stations$station[[1]]$longitude,
#   start = "2024-01-01",
#   end = "2024-01-31",
#   daily = c(
#     # "temperature_2m_max",
#     # "temperature_2m_min",
#     # "precipitation_sum",
#     # "wind_speed_10m_max",
#     # "shortwave_radiation_sum",
#     # "et0_fao_evapotranspiration"
#   )
# )

# stations_met <- stations %>%
#   mutate(
#     met_start = min(as_date(images_start)) - days(95),
#     met_end = max(as_date(images_end)) + days(2),
#     met_hr = list({
#       get_met(
#         location = c(station$latitude, station$longitude),
#         start = as.character(met_start),
#         end = as.character(met_end),
#         hourly = c(
#           "temperature_2m",
#           "relative_humidity_2m",
#           "rain",
#           "snowfall",
#           "snow_depth",
#           "et0_fao_evapotranspiration",
#           "vapour_pressure_deficit",
#           "soil_moisture_0_to_7cm"
#         ),
#         timezone = "UTC"
#       )
#     }),
#     met_day = list({
#       get_met(
#         location = c(station$latitude, station$longitude),
#         start = as.character(met_start),
#         end = as.character(met_end),
#         daily = c(
#           "temperature_2m",
#           "daylight_duration",
#           "rain_sum",
#           "snowfall_sum",
#           "shortwave_radiation_sum",
#           "et0_fao_evapotranspiration",
#           "soil_moisture_0_to_100cm_mean"
#         )
#       )
#     })
#   )
# write_rds(stations_met, "cache/stations-met.rds")
stations_met <- read_rds("cache/stations-met.rds")

stopifnot(all(stations$station_id %in% stations_met$station_id))

# aux ---------------------------------------------------------------------

transform_aux_day <- function (met_day) {
  met_day %>%
    transmute(
      date,
      airtemp = daily_temperature_2m_mean,
      daylight = daily_daylight_duration,
      rain = daily_rain_sum,
      snowfall = daily_snowfall_sum,
      srad = daily_shortwave_radiation_sum,
      et = daily_et0_fao_evapotranspiration,
      soilmoisture = daily_soil_moisture_0_to_100cm_mean,
    ) %>%
    mutate(
      across(-date, \(x) (x - min(x)) / (max(x) - min(x)))
    )
}
compute_aux_day_stats <- function (aux_day) {
  x_lag <- aux_day %>%
    pivot_longer(-date) %>%
    nest_by(name) %>%
    mutate(
      name = glue("{name}_lag1d"),
      data = list({
        data %>%
          mutate(
            value = lag(value, n = 1)
          )
      })
    ) %>%
    ungroup() %>%
    unnest(data) %>%
    pivot_wider()
  x_mean <- aux_day %>%
    select(-rain, -snowfall) %>%
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
  x_sum <- aux_day %>%
    select(date, rain, snowfall) %>%
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
  x_other <- aux_day %>%
    transmute(
      date,
      rain_lag2d = lag(rain, n = 2),
      rain_lag3d = lag(rain, n = 3),
      rain_lag4d = lag(rain, n = 4),
      rain_lag5d = lag(rain, n = 5),
      rain_lag6d = lag(rain, n = 6),
      rain_lag7d = lag(rain, n = 7),

      rain_max7d = na_if(slider::slide_dbl(rain, .before = 7, .after = -1, .f = max), -Inf),
      rain_max30d = na_if(slider::slide_dbl(rain, .before = 30, .after = -1, .f = max), -Inf),

      rain_sd7d = slider::slide_dbl(rain, .before = 7, .after = -1, .f = sd),
      rain_sd30d = slider::slide_dbl(rain, .before = 30, .after = -1, .f = sd),

      snowfall_max7d = na_if(slider::slide_dbl(snowfall, .before = 7, .after = -1, .f = max), -Inf),
      snowfall_max30d = na_if(slider::slide_dbl(snowfall, .before = 30, .after = -1, .f = max), -Inf),

      airtemp_lag2d = lag(airtemp, n = 2),
      airtemp_lag3d = lag(airtemp, n = 3),

      airtemp_max7d = na_if(slider::slide_dbl(airtemp, .before = 7, .after = -1, .f = max), -Inf),
      airtemp_max30d = na_if(slider::slide_dbl(airtemp, .before = 30, .after = -1, .f = max), -Inf),

      airtemp_min7d = na_if(slider::slide_dbl(airtemp, .before = 7, .after = -1, .f = min), Inf),
      airtemp_min30d = na_if(slider::slide_dbl(airtemp, .before = 30, .after = -1, .f = min), Inf),
    )
  x_lag %>%
    left_join(x_mean, by = "date") %>%
    left_join(x_sum, by = "date") %>%
    left_join(x_other, by = "date") %>%
    mutate(
      across(-date, ~ data.table::nafill(., type = "nocb"))
    )
}
# compute_aux_day_stats(transform_aux_day(stations_met$met_day[[1]]))

transform_aux_hr <- function (met_hr) {
  met_hr %>%
    transmute(
      timestamp = datetime,
      airtemp = hourly_temperature_2m,
      relhum = hourly_relative_humidity_2m,
      rain = hourly_rain,
      snowfall = hourly_snowfall,
      snow_depth = coalesce(hourly_snow_depth, 0),
      et = hourly_et0_fao_evapotranspiration,
      vpd = hourly_vapour_pressure_deficit,
      soilmoisture = hourly_soil_moisture_0_to_7cm
    ) %>%
    mutate(
      across(-timestamp, function (x) {
        if (max(x) == min(x)) return(x)
        (x - min(x)) / (max(x) - min(x))
      })
    )
}
compute_aux_hr_stats <- function (aux_hr) {
  x_lag <- aux_hr %>%
    pivot_longer(-timestamp) %>%
    crossing(
      n = c(1, 2, 3, 4)
    ) %>%
    nest_by(name, n) %>%
    mutate(
      name = glue("{name}_lag{n}d"),
      data = list({
        data %>%
          mutate(
            value = lag(value, n = n)
          )
      })
    ) %>%
    ungroup() %>%
    select(-n) %>%
    unnest(data) %>%
    pivot_wider()
  x_mean <- aux_hr %>%
    select(-rain, -snowfall) %>%
    pivot_longer(-timestamp) %>%
    crossing(
      ndays = c(1, 3, 7, 30, 90)
    ) %>%
    nest_by(name, ndays) %>%
    mutate(
      name = glue("{name}_mean{ndays}d"),
      data = list({
        data %>%
          mutate(
            value = slider::slide_dbl(value, .before = ndays * 24, .after = -1, .f = mean)
          )
      })
    ) %>%
    ungroup() %>%
    select(-ndays) %>%
    unnest(data) %>%
    pivot_wider()
  x_sum <- aux_hr %>%
    select(timestamp, rain, snowfall) %>%
    pivot_longer(-timestamp) %>%
    crossing(
      ndays = c(1, 3, 7, 30, 90)
    ) %>%
    nest_by(name, ndays) %>%
    mutate(
      name = glue("{name}_sum{ndays}d"),
      data = list({
        data %>%
          mutate(
            value = slider::slide_dbl(value, .before = ndays * 24, .after = -1, .f = sum)
          )
      })
    ) %>%
    ungroup() %>%
    select(-ndays) %>%
    unnest(data) %>%
    pivot_wider()
  x_other <- aux_hr %>%
    transmute(
      timestamp,

      rain_max1d = na_if(slider::slide_dbl(rain, .before = 1 * 24, .after = -1, .f = max), -Inf),
      rain_max3d = na_if(slider::slide_dbl(rain, .before = 3 * 24, .after = -1, .f = max), -Inf),
      rain_max7d = na_if(slider::slide_dbl(rain, .before = 7 * 25, .after = -1, .f = max), -Inf),
      rain_max30d = na_if(slider::slide_dbl(rain, .before = 30 * 24, .after = -1, .f = max), -Inf),

      rain_sd1d = slider::slide_dbl(rain, .before = 1 * 24, .after = -1, .f = max),
      rain_sd3d = slider::slide_dbl(rain, .before = 3 * 24, .after = -1, .f = max),
      rain_sd7d = slider::slide_dbl(rain, .before = 7 * 24, .after = -1, .f = sd),
      rain_sd30d = slider::slide_dbl(rain, .before = 30 * 24, .after = -1, .f = sd),

      snowfall_max7d = na_if(slider::slide_dbl(snowfall, .before = 7 * 24, .after = -1, .f = max), -Inf),
      snowfall_max30d = na_if(slider::slide_dbl(snowfall, .before = 30 * 24, .after = -1, .f = max), -Inf),

      airtemp_max1d = na_if(slider::slide_dbl(airtemp, .before = 1 * 24, .after = -1, .f = max), -Inf),
      airtemp_max3d = na_if(slider::slide_dbl(airtemp, .before = 3 * 24, .after = -1, .f = max), -Inf),
      airtemp_max7d = na_if(slider::slide_dbl(airtemp, .before = 7 * 24, .after = -1, .f = max), -Inf),
      airtemp_max30d = na_if(slider::slide_dbl(airtemp, .before = 30 * 24, .after = -1, .f = max), -Inf),

      airtemp_min1d = na_if(slider::slide_dbl(airtemp, .before = 1 * 24, .after = -1, .f = min), Inf),
      airtemp_min3d = na_if(slider::slide_dbl(airtemp, .before = 3 * 24, .after = -1, .f = min), Inf),
      airtemp_min7d = na_if(slider::slide_dbl(airtemp, .before = 7 * 24, .after = -1, .f = min), Inf),
      airtemp_min30d = na_if(slider::slide_dbl(airtemp, .before = 30 * 24, .after = -1, .f = min), Inf)
    )
  x_lag %>%
    left_join(x_mean, by = "timestamp") %>%
    left_join(x_sum, by = "timestamp") %>%
    left_join(x_other, by = "timestamp") %>%
    mutate(
      across(-timestamp, ~ data.table::nafill(., type = "nocb"))
    )
}

stations_met %>%
  select(station_code, met_day) %>%
  unnest(met_day) %>%
  pivot_longer(-c(station_code, date)) %>%
  ggplot(aes(date, value)) +
  geom_line() +
  facet_grid(vars(name), vars(station_code), scales = "free_y")

stations_aux <- stations_met %>%
  transmute(
    station_id,
    aux_day = list(transform_aux_day(met_day)),
    aux_day_stats = list(compute_aux_day_stats(aux_day)),
    aux_hr = list(transform_aux_hr(met_hr)),
    aux_hr_stats = list(compute_aux_hr_stats(aux_hr))
  )

# inputs --------------------------------------------------------------

n_train_pairs <- 400
n_val_pairs <- 100

set.seed(2119)
stations_inp <- stations %>%
  mutate(
    pairs = list({
      x_train <- pairs %>%
        filter(file_downloaded_1 & file_downloaded_2) %>%
        nest_by(pair) %>%
        ungroup() %>%
        slice_sample(n = n_train_pairs) %>%
        unnest(data) %>%
        mutate(split = "train")
      x_val <- pairs %>%
        filter(file_downloaded_1 & file_downloaded_2) %>%
        filter(!pair %in% x_train$pair) %>%
        nest_by(pair) %>%
        ungroup() %>%
        slice_sample(n = n_val_pairs) %>%
        unnest(data) %>%
        mutate(split = "val")

      bind_rows(x_train, x_val)
    }),
    images = list({
      x_train_pairs <- pairs %>%
        filter(split == "train")
      x_val_pairs <- pairs %>%
        filter(split == "val")
      images %>%
        mutate(
          split = case_when(
            date >= test_start ~ "test-out",
            image_id %in% c(x_train_pairs$image_id_1, x_train_pairs$image_id_2) ~ "train",
            image_id %in% c(x_val_pairs$image_id_1, x_val_pairs$image_id_2) ~ "val",
            TRUE ~ "test-in"
          )
        ) %>%
        nest_by(split) %>%
        mutate(
          data = list({
            if (split %in% c("train", "val")) {
              data
            } else {
              n_total <- 1000
              data_downloaded <- data %>%
                filter(file_downloaded)
              if (nrow(data_downloaded) > n_total) {
                data_downloaded <- data_downloaded %>%
                  slice_sample(n = n_total)
              }
              n_downloaded <- nrow(data_downloaded)
              n_remaining <- n_total - n_downloaded
              data_remaining <- data %>%
                filter(!file_downloaded) %>%
                slice_sample(n = n_remaining)
              bind_rows(data_downloaded, data_remaining)
            }
          })
        ) %>%
        unnest(data)
    }),
    images_n = nrow(images),
    images_n_downloaded = sum(images$file_downloaded),
    images_n_missing = sum(!images$file_downloaded),
    pairs_n = nrow(pairs) / 2,
    pairs_n_downloaded = sum(pairs$file_downloaded_1 & pairs$file_downloaded_2) / 2,
    pairs_n_missing = sum(!(pairs$file_downloaded_1 & pairs$file_downloaded_2)) / 2,
  ) %>%
  left_join(
    stations_aux %>%
      select(station_id, starts_with("aux")),
    by = "station_id"
  )

summary(stations_inp$pairs_n)
summary(stations_inp$images_n)
stations_inp %>%
  select(pairs_n, pairs_n_downloaded, pairs_n_missing)
stations_inp %>%
  select(images_n, images_n_downloaded, images_n_missing)

# export list of images to download
# then run download-images.sh
stations_inp %>%
  select(images) %>%
  unnest(images) %>%
  filter(!file_downloaded) %>%
  pull(filename) %>%
  write_lines("images.txt")

# ts: n pairs and n images by month
# stations %>%
#   select(station_code, pairs) %>%
#   unnest(pairs) %>%
#   count(station_code, date = floor_date(date_1, "month"), name = "n_pairs") %>%
#   full_join(
#     stations %>%
#       select(station_code, images) %>%
#       unnest(images) %>%
#       count(station_code, date = floor_date(date, "month"), name = "n_images")
#   ) %>%
#   ggplot(aes(date, n_pairs)) +
#   geom_point() +
#   geom_line() +
#   geom_point(aes(y = n_images), color = "deepskyblue") +
#   geom_line(aes(y = n_images), color = "deepskyblue") +
#   facet_wrap(vars(station_code), scales = "free")

# ts: images, color=split
stations_inp %>%
  select(station_code, images) %>%
  unnest(images) %>%
  ggplot(aes(timestamp, value)) +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  facet_wrap(vars(station_code), scales = "free")

# check that all images have been downloaded
stations_inp %>%
  select(images) %>%
  unnest(images) %>%
  mutate(
    exists = map_lgl(filename, ~file.exists(file.path("~/data/fpe/images", .)))
  ) %>%
  group_by(station_id, station_name) %>%
  summarise(
    n_exists = sum(exists),
    n_missing = n() - n_exists,
    frac_exists = mean(exists)
  )

# splot: pair values, color=label
stations_inp %>%
  select(station_code, pairs) %>%
  unnest(pairs) %>%
  ggplot(aes(value_1, value_2, color = factor(label))) +
  geom_point(size = 0.5, alpha = 0.5) +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(station_code), scales = "free") +
  theme(aspect.ratio = 1)

# splot: pair values, color=split
stations_inp %>%
  select(station_code, pairs) %>%
  unnest(pairs) %>%
  ggplot(aes(value_1, value_2, color = factor(split))) +
  geom_point(size = 0.5, alpha = 0.5) +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(station_code), scales = "free") +
  theme(aspect.ratio = 1)

# pairs: split counts
stations_inp %>%
  select(station_code, pairs) %>%
  unnest(pairs) %>%
  tabyl(station_code, split)

# images: split counts
stations_inp %>%
  select(station_code, images) %>%
  unnest(images) %>%
  tabyl(station_code, split)

# runs --------------------------------------------------------------------

# runs/<model>/<station_id>

models <- tribble(
  ~model,      ~aux_model, ~aux_timestep, ~aux_dataset,
  "none",      NA,         NA,            NA,
  "concat-d",  "concat",   "D",           "aux_day_stats",
  "concat-h",  "concat",   "H",           "aux_hr_stats",
  "encoder-d", "encoder",  "D",           "aux_day_stats",
  "encoder-h", "encoder",  "H",           "aux_hr_stats",
  "lstm-d",    "lstm",     "D",           "aux_day",
  "lstm-h",    "lstm",     "H",           "aux_hr",
)

runs <- models %>%
  crossing(
    stations_inp %>%
      select(station_id, images, pairs, aux_day, aux_day_stats, aux_hr, aux_hr_stats)
  ) %>%
  rowwise() %>%
  mutate(
    run_name = glue("{model}_{station_id}"),
    run_dir = {
      d <- file.path("runs", model, station_id)
      dir.create(file.path(d, "input"), showWarnings = FALSE, recursive = TRUE)
      d
    }
  )
stopifnot(all(!duplicated(runs$run_name)))

for (i in 1:nrow(runs)) {
  run <- runs[i, ]
  input_dir <- file.path(run$run_dir, "input")
  run$images[[1]] %>%
    write_csv(file.path(input_dir, "images.csv"))
  run$pairs[[1]] %>%
    write_csv(file.path(input_dir, "pairs.csv"))
  config <- list(
    mlflow_experiment_name = experiment,
    mlflow_run_name = run$run_name,
    random_seed = 1210L
  )
  if (!is.na(run$aux_model)) {
    aux_file <- glue("{run$aux_dataset}.csv")
    run[[run$aux_dataset]][[1]] %>%
      write_csv(file.path(input_dir, aux_file))

    config <- c(config, list(
      aux_model = run$aux_model,
      aux_timestep = run$aux_timestep,
      aux_file = aux_file
    ))
    if (run$model == "lstm-d") {
      config[["aux_lstm_sequence_length"]] <- 30L
    } else if (run$model == "lstm-h") {
      config[["aux_lstm_sequence_length"]] <- as.integer(7L * 24L)
    }
  }

  config %>%
    write_yaml(file.path(input_dir, "config.yml"))
}


# results -----------------------------------------------------------------

pred <- runs %>%
  left_join(select(stations, station_id, station_code), by = "station_id") %>%
  select(model, station_id, station_code) %>%
  mutate(
    data = list({
      f <- file.path("runs", model, station_id, "output", "data", "predictions.csv")
      if (file.exists(f)) {
        read_csv(f, show_col_types = FALSE)
      } else {
        tibble()
      }
    })
  ) %>%
  mutate(model = factor(model, levels = models$model)) %>%
  filter(nrow(data) > 0) %>%
  select(-station_id)

# splot: by model
pred %>%
  unnest(data) %>%
  # filter(station_id %in% c(12, 13)) %>%
  group_by(model, station_code) %>%
  mutate(across(c(value, prediction), scale)) %>%
  ungroup() %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  facet_grid(vars(station_code), vars(model)) +
  theme(aspect.ratio = 1)

# splot: by split
pred %>%
  unnest(data) %>%
  group_by(model, station_code) %>%
  mutate(across(c(value, prediction), scale)) %>%
  ungroup() %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = model), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_grid(vars(split), vars(station_code)) +
  theme(aspect.ratio = 1)

# tau: table
pred %>%
  unnest(data) %>%
  group_by(model, station_code, split) %>%
  summarise(
    tau = cor(value, prediction, method = "kendall"),
    .groups = "drop"
  ) %>%
  pivot_wider(names_from = "model", values_from = "tau")

# tau: dot plot by split, color=model
pred %>%
  unnest(data) %>%
  bind_rows(
    pred %>%
      unnest(data) %>%
      mutate(split = "all")
  ) %>%
  mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out", "all"))) %>%
  group_by(model, station_code, split) %>%
  summarise(
    tau = cor(value, prediction, method = "kendall"),
    .groups = "drop"
  ) %>%
  ggplot(aes(station_code, tau)) +
  geom_point(aes(color = model)) +
  coord_flip() +
  facet_grid(vars(split))

# tau: box plot by split, color=model
pred %>%
  unnest(data) %>%
  bind_rows(
    pred %>%
      unnest(data) %>%
      mutate(split = "all")
  ) %>%
  mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out", "all"))) %>%
  group_by(model, station_code, split) %>%
  summarise(
    tau = cor(value, prediction, method = "kendall"),
    .groups = "drop"
  ) %>%
  ggplot(aes(fct_rev(model), tau)) +
  geom_boxplot(aes()) +
  geom_jitter(aes(color = model), height = 0, width = 0.2) +
  coord_flip() +
  facet_wrap(vars(split), ncol = 1)


# tau: splotmat by model
# - encoder-d does more poorly than none and concat-d on test-out
pred %>%
  unnest(data) %>%
  bind_rows(
    pred %>%
      unnest(data) %>%
      mutate(split = "all")
  ) %>%
  mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out", "all"))) %>%
  group_by(model, station_code, split) %>%
  summarise(
    tau = cor(value, prediction, method = "kendall"),
    .groups = "drop"
  ) %>%
  pivot_wider(names_from = "model", values_from = "tau") %>%
  select(-station_code) %>%
  GGally::ggpairs(mapping = aes(color = split)) +
  geom_abline()

# tau: splotmat by split
# - encoder-d does more poorly than none and concat-d on test-out
pred %>%
  unnest(data) %>%
  bind_rows(
    pred %>%
      unnest(data) %>%
      mutate(split = "all")
  ) %>%
  mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out", "all"))) %>%
  group_by(model, station_code, split) %>%
  summarise(
    tau = cor(value, prediction, method = "kendall"),
    .groups = "drop"
  ) %>%
  pivot_wider(names_from = "split", values_from = "tau") %>%
  select(-station_code) %>%
  GGally::ggpairs(mapping = aes(color = model)) +
  geom_abline()

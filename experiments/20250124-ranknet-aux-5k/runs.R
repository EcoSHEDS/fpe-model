setwd("~/git/fpe-model/experiments/20250124-ranknet-aux-5k/")

library(tidyverse)
library(jsonlite)
library(janitor)
library(glue)
library(daymetr)
library(openmeteo)
library(yaml)
library(patchwork)
library(gt)

experiment <- "20250124-ranknet-aux-5k"
daytime_hours <- 8:16

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
#   hourly = c(
#     "temperature_2m",
#     "relative_humidity_2m",
#     "pressure_msl",
#     "wind_speed_10m",
#     "shortwave_radiation",
#     "cloud_cover",
#     "rain",
#     "snowfall",
#     "et0_fao_evapotranspiration",
#     "vapour_pressure_deficit",
#     "soil_moisture_0_to_7cm",
#     "soil_moisture_100_to_255cm"
#   ),
#   timezone = "UTC"
# )

# stations_met_raw <- stations %>%
#   mutate(
#     met_start = min(as_date(images_start)) - years(2),
#     met_end = max(as_date(images_end)) + days(10),
#     met_hr = list({
#       get_met(
#         location = c(station$latitude, station$longitude),
#         start = as.character(met_start),
#         end = as.character(met_end),
#         hourly = c(
#           "temperature_2m",
#           "relative_humidity_2m",
#           "pressure_msl",
#           "wind_speed_10m",
#           "shortwave_radiation",
#           "cloud_cover",
#           "rain",
#           "snowfall",
#           "et0_fao_evapotranspiration",
#           "vapour_pressure_deficit",
#           "soil_moisture_0_to_7cm",
#           "soil_moisture_100_to_255cm"
#         ),
#         timezone = "UTC"
#       )
#     })
#   )
# write_rds(stations_met_raw, "cache/stations-met-raw.rds")
stations_met_raw <- read_rds("cache/stations-met-raw.rds")

stations_met <- stations_met_raw %>%
  left_join(
    stations %>%
      rowwise() %>%
      mutate(timezone = station$timezone) %>%
      select(station_id, timezone),
    by = "station_id"
  ) %>%
  mutate(
    met_hr = list({
      met_hr %>%
        rename_with(.cols = -datetime, .fn = \(x) str_remove(x, "hourly_")) %>%
        mutate(
          datetime_local = with_tz(datetime, tz = timezone),
          date = as_date(datetime_local),
          .after = datetime
        )
    }),
    met_day = list({
      met_hr %>%
        group_by(date) %>%
        summarise(
          n = n(),
          across(
            c(temperature_2m),
            .fns = c(min = min, mean = mean, max = max),
            .names = "{.col}_{.fn}"
          ),
          across(
            c(rain),
            .fns = c(sum = sum, max = max),
            .names = "{.col}_{.fn}"
          ),
          across(
            c(snowfall, et0_fao_evapotranspiration),
            .fns = c(sum = sum),
            .names = "{.col}_{.fn}"
          ),
          across(
            c(relative_humidity_2m, pressure_msl, wind_speed_10m,
              shortwave_radiation, cloud_cover, vapour_pressure_deficit,
              soil_moisture_0_to_7cm, soil_moisture_100_to_255cm),
            .fns = c(mean = mean),
            .names = "{.col}_{.fn}"
          )
        ) %>%
        filter(n >= 23) %>%
        select(-n)
    })
  )

# check timezone based on diurnal air temperature
# max should be 3pm (1500)
stations_met %>%
  select(station_code, met_hr) %>%
  unnest(met_hr) %>%
  ggplot(aes(factor(hour(datetime_local)), temperature_2m)) +
  geom_boxplot()

stopifnot(all(stations$station_id %in% stations_met$station_id))

# aux ---------------------------------------------------------------------

transform_aux_day <- function (met_day) {
  met_day %>%
    transmute(
      date,
      airtemp_min = temperature_2m_min,
      airtemp_mean = temperature_2m_mean,
      airtemp_max = temperature_2m_max,
      rain_sum = rain_sum,
      rain_max = rain_max,
      snowfall_sum = snowfall_sum,
      et_sum = et0_fao_evapotranspiration_sum,
      rh_mean = relative_humidity_2m_mean,
      pressure_mean = pressure_msl_mean,
      windspd_mean = wind_speed_10m_mean,
      srad_mean = shortwave_radiation_mean,
      cloud_mean = cloud_cover_mean,
      vpd_mean = vapour_pressure_deficit_mean,
      soilmtop_mean = soil_moisture_0_to_7cm_mean,
      soilmbot_mean = soil_moisture_100_to_255cm_mean
    ) %>%
    mutate(
      across(-date, \(x) (x - min(x)) / (max(x) - min(x)))
    )
}
# aux_day <- transform_aux_day(stations_met$met_day[[1]])

compute_aux_day_stats <- function (aux_day) {
  x_current <- aux_day %>%
    select(-ends_with("_min"), -ends_with("_max")) %>%
    pivot_longer(-date)
  x_lag <- aux_day %>%
    select(-starts_with("soilm"), -ends_with("_min"), -ends_with("_max")) %>%
    pivot_longer(-date) %>%
    crossing(
      n_day = c(1, 2, 3)
    ) %>%
    nest_by(name, n_day) %>%
    mutate(
      name = glue("{name}_lag{n_day}d"),
      data = list({
        data %>%
          mutate(
            value = lag(value, n = n_day)
          )
      })
    ) %>%
    ungroup() %>%
    select(-n_day) %>%
    unnest(data)
  x_mean <- aux_day %>%
    select(-rain_sum, -rain_max, -snowfall_sum, -starts_with("soilm"), -ends_with("_min"), -ends_with("_max")) %>%
    pivot_longer(-date) %>%
    crossing(
      n_day = c(7, 30, 60, 90)
    ) %>%
    nest_by(name, n_day) %>%
    mutate(
      name = glue("{name}_mean{n_day}d"),
      data = list({
        data %>%
          mutate(
            value = slider::slide_dbl(value, .before = n_day, .after = -1, .f = mean)
          )
      })
    ) %>%
    ungroup() %>%
    select(-n_day) %>%
    unnest(data)
  x_sum <- aux_day %>%
    select(date, rain_sum, snowfall_sum) %>%
    pivot_longer(-date) %>%
    crossing(
      n_day = c(7, 30, 60, 90)
    ) %>%
    nest_by(name, n_day) %>%
    mutate(
      name = glue("{name}_sum{n_day}d"),
      data = list({
        data %>%
          mutate(
            value = slider::slide_dbl(value, .before = n_day, .after = -1, .f = sum)
          )
      })
    ) %>%
    ungroup() %>%
    select(-n_day) %>%
    unnest(data)
  x_max <- aux_day %>%
    select(date, rain_max, airtemp_max) %>%
    pivot_longer(-date) %>%
    crossing(
      n_day = c(7, 30)
    ) %>%
    nest_by(name, n_day) %>%
    mutate(
      name = glue("{name}_max{n_day}d"),
      data = list({
        data %>%
          mutate(
            value = na_if(slider::slide_dbl(value, .before = n_day, .after = -1, .f = max), -Inf)
          )
      })
    ) %>%
    ungroup() %>%
    select(-n_day) %>%
    unnest(data)
  x_min <- aux_day %>%
    select(date, airtemp_min) %>%
    pivot_longer(-date) %>%
    crossing(
      n_day = c(7, 30)
    ) %>%
    nest_by(name, n_day) %>%
    mutate(
      name = glue("{name}_min{n_day}d"),
      data = list({
        data %>%
          mutate(
            value = na_if(slider::slide_dbl(value, .before = n_day, .after = -1, .f = min), Inf)
          )
      })
    ) %>%
    ungroup() %>%
    select(-n_day) %>%
    unnest(data)
  x_sd <- aux_day %>%
    select(date, rain_sum) %>%
    pivot_longer(-date) %>%
    crossing(
      n_day = c(7, 30)
    ) %>%
    nest_by(name, n_day) %>%
    mutate(
      name = glue("{name}_sd{n_day}d"),
      data = list({
        data %>%
          mutate(
            value = slider::slide_dbl(value, .before = n_day, .after = -1, .f = sd)
          )
      })
    ) %>%
    ungroup() %>%
    select(-n_day) %>%
    unnest(data)
  bind_rows(
    x_current,
    x_lag,
    x_mean,
    x_sum,
    x_min,
    x_max,
    x_sd
  ) %>%
    pivot_wider() %>%
    mutate(
      across(-date, ~ data.table::nafill(., type = "nocb"))
    )
}
# aux_day_stats <- compute_aux_day_stats(transform_aux_day(stations_met$met_day[[1]]))

aux_day_stats %>%
  select(-date) %>%
  cor() %>%
  as_tibble(rownames = "var1") %>%
  pivot_longer(-var1, names_to = "var2") %>%
  filter(var1 != var2) %>%
  # group_by(var1) %>%
  # summarise(value = median(abs(value))) %>%
  # arrange(desc(value))
  ggplot(aes(value)) +
  geom_histogram(binwidth = 0.05)

transform_aux_hr <- function (met_hr) {
  met_hr %>%
    transmute(
      timestamp = datetime,
      airtemp = temperature_2m,
      rh = relative_humidity_2m,
      pressure = pressure_msl,
      windspd = wind_speed_10m,
      srad = shortwave_radiation,
      cloud = cloud_cover,
      rain = rain,
      snowfall = snowfall,
      et = et0_fao_evapotranspiration,
      vpd = vapour_pressure_deficit,
      soilmtop = soil_moisture_0_to_7cm,
      soilmbot = soil_moisture_100_to_255cm
    ) %>%
    mutate(
      across(-timestamp, function (x) {
        if (max(x) == min(x)) return(x)
        (x - min(x)) / (max(x) - min(x))
      })
    )
}
# aux_hr <- transform_aux_hr(stations_met$met_hr[[1]])

compute_aux_hr_stats <- function (aux_hr) {
  x_current <- aux_hr %>%
    pivot_longer(-timestamp)
  x_lag <- aux_hr %>%
    select(timestamp, rain) %>%
    pivot_longer(-timestamp) %>%
    crossing(
      n_hr = 1:6
    ) %>%
    nest_by(name, n_hr) %>%
    mutate(
      name = glue("{name}_lag{n_hr}hr"),
      data = list({
        data %>%
          mutate(
            value = lag(value, n = n_hr)
          )
      })
    ) %>%
    ungroup() %>%
    select(-n_hr) %>%
    unnest(data)
  x_sum <- aux_hr %>%
    select(timestamp, rain, snowfall) %>%
    pivot_longer(-timestamp) %>%
    crossing(
      n_hr = c(24, 48, 72)
    ) %>%
    nest_by(name, n_hr) %>%
    mutate(
      name = glue("{name}_sum{n_hr}hr"),
      data = list({
        data %>%
          mutate(
            value = slider::slide_dbl(value, .before = n_hr, .after = -1, .f = sum)
          )
      })
    ) %>%
    ungroup() %>%
    select(-n_hr) %>%
    unnest(data)
  x_max <- aux_hr %>%
    select(timestamp, rain) %>%
    pivot_longer(-timestamp) %>%
    crossing(
      n_hr = c(24, 48, 72)
    ) %>%
    nest_by(name, n_hr) %>%
    mutate(
      name = glue("{name}_max{n_hr}hr"),
      data = list({
        data %>%
          mutate(
            value = na_if(slider::slide_dbl(value, .before = n_hr, .after = -1, .f = max), -Inf)
          )
      })
    ) %>%
    ungroup() %>%
    select(-n_hr) %>%
    unnest(data)
  x_sd <- aux_hr %>%
    select(timestamp, rain) %>%
    pivot_longer(-timestamp) %>%
    crossing(
      n_hr = c(24, 48, 72)
    ) %>%
    nest_by(name, n_hr) %>%
    mutate(
      name = glue("{name}_sd{n_hr}hr"),
      data = list({
        data %>%
          mutate(
            value = slider::slide_dbl(value, .before = n_hr, .after = -1, .f = sd)
          )
      })
    ) %>%
    ungroup() %>%
    select(-n_hr) %>%
    unnest(data)

  bind_rows(
    x_lag,
    x_sum,
    x_max,
    x_sd
  ) %>%
    pivot_wider() %>%
    mutate(
      across(-timestamp, ~ data.table::nafill(., type = "nocb"))
    )
}
aux_hr_stats <- compute_aux_hr_stats(transform_aux_hr(stations_met$met_hr[[1]]))

aux_hr_stats %>%
  select(-timestamp) %>%
  cor() %>%
  as_tibble(rownames = "var1") %>%
  pivot_longer(-var1, names_to = "var2") %>%
  filter(var1 != var2, var1 < var2) %>%
  # group_by(var1) %>%
  # summarise(value = median(abs(value))) %>%
  # mutate(value = abs(value)) %>%
  # arrange(desc(value)) %>% print(n = 20)
  ggplot(aes(value)) +
  geom_histogram(binwidth = 0.05)

stations_met %>%
  select(station_code, met_day) %>%
  unnest(met_day) %>%
  pivot_longer(-c(station_code, date)) %>%
  ggplot(aes(date, value)) +
  geom_line() +
  facet_grid(vars(name), vars(station_code), scales = "free_y")

stations_met %>%
  head(1) %>%
  select(station_code, met_day) %>%
  unnest(met_day) %>%
  pivot_longer(-c(station_code, date)) %>%
  ggplot(aes(date, value)) +
  geom_line() +
  labs(x = NULL, y = NULL) +
  facet_wrap(vars(name), scales = "free_y", ncol = 3) +
  theme_bw()


stations_aux <- stations_met %>%
  transmute(
    station_id,
    aux_day = list(transform_aux_day(met_day)),
    aux_day_stats = list(compute_aux_day_stats(aux_day)),
    aux_hr = list(transform_aux_hr(met_hr)),
    aux_hr_stats = list(compute_aux_hr_stats(aux_hr)),
    aux_stats = list({
      aux_hr_stats %>%
        mutate(
          date = as_date(with_tz(timestamp, tz = timezone)),
          .before = everything()
        ) %>%
        inner_join(aux_day_stats, by = "date") %>%
        select(-date)
    })
  )
stopifnot(
  all(!is.na(stations_aux$aux_stats[[1]])),
  all(!is.na(stations_aux$aux_hr[[1]])),
  all(!is.na(stations_aux$aux_day[[1]]))
)

stations_aux$aux_day_stats[[1]] %>%
  select(-date) %>%
  cor() %>%
  as_tibble(rownames = "var1") %>%
  pivot_longer(-var1, names_to = "var2") %>%
  filter(var1 < var2) %>%
  # group_by(var1) %>%
  # summarise(value = median(abs(value))) %>%
  # mutate(value = abs(value)) %>%
  # arrange(desc(value)) %>%
  mutate(high = abs(value) > 0.8) %>% pull(high) %>% mean()
  ggplot(aes(value)) +
  geom_histogram(binwidth = 0.05)


# annotator ---------------------------------------------------------------



# inputs --------------------------------------------------------------

n_train_pairs <- 4000
n_val_pairs <- 1000

set.seed(1151)
stations_inp <- stations %>%
  mutate(
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
              data %>%
                slice_sample(n = n_total)
            }
          })
        ) %>%
        unnest(data)
    }),
    images_n = nrow(images),
    pairs_n = nrow(pairs) / 2
  ) %>%
  left_join(
    stations_aux %>%
      select(station_id, starts_with("aux")),
    by = "station_id"
  )

summary(stations_inp$pairs_n)
summary(stations_inp$images_n)

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
  ~model,       ~aux_model, ~aux_timestep, ~aux_dataset, ~lstm_len,
  "none",       NA,         NA,            NA,           NA,
  "concat",     "concat",   "H",           "aux_stats",  NA,
  "encoder",    "encoder",  "H",           "aux_stats",  NA,
  # "lstm-d-30",  "lstm",     "D",           "aux_day",    30,
  "lstm-d-90",  "lstm",     "D",           "aux_day",    90,
  # "lstm-d-180", "lstm",     "D",           "aux_day",    180,
  "lstm-h-30",  "lstm",     "H",           "aux_hr",     30 * 24,
  # "lstm-h-60",  "lstm",     "H",           "aux_hr",     60 * 24,
  # "lstm-h-90",  "lstm",     "H",           "aux_hr",     90 * 24
)

runs <- models %>%
  crossing(
    stations_inp %>%
      select(station_id, images, pairs, aux_day, aux_hr, aux_stats)
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
    if (!is.na(run$lstm_len)) {
      config[["aux_lstm_sequence_length"]] <- as.integer(run$lstm_len)
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
  select(-station_id) %>%
  mutate(
    model = case_when(
      model == "none" ~ "no aux",
      model == "concat" ~ "direct",
      model == "encoder" ~ "fcn",
      model == "lstm-d-90" ~ "lstm-day",
      model == "lstm-h-30" ~ "lstm-hour"
    ),
    model = factor(model, levels = c("no aux", "direct", "fcn", "lstm-day", "lstm-hour"))
  )
list(
  stations = stations,
  stations_inp = stations_inp,
  pred = pred
) %>%
  write_rds("cache/runs.rds")

pred <- read_rds("cache/runs.rds")$pred

pred_tau <- pred %>%
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
    rmse = sqrt(mean((scale(value) - scale(prediction)) ^ 2)),
    .groups = "drop"
  ) %>%
  ungroup()

## memo figures ------------------------------------------------------------

p <- pred_tau %>%
  ggplot(aes(fct_rev(model), tau)) +
  geom_hline(yintercept = 0, alpha = 0.5) +
  geom_boxplot(fill = "lightgrey") +
  coord_flip() +
  labs(x = "model") +
  facet_wrap(vars(split), ncol = 2) +
  theme_bw()
ggsave("figs/tau-box-split.png", plot = p, width = 6, height = 6)

pred_tau %>%
  filter(split == "all") %>%
  group_by(model) %>%
  summarise(tau = median(tau)) %>%
  arrange(tau)

pred_tau %>%
  filter(split == "test-out") %>%
  group_by(model) %>%
  summarise(tau = median(tau)) %>%
  arrange(tau)

pred_tau %>%
  group_by(model, split) %>%
  summarise(tau = median(tau), .groups = "drop") %>%
  pivot_wider(names_from = "split", values_from = "tau")

p <- pred_tau %>%
  ggplot(aes(model, fct_rev(split))) +
  geom_tile(aes(fill = tau)) +
  geom_text(aes(label = sprintf("%.2f", tau)), size = 3) +
  scale_fill_distiller(palette = "YlGnBu", limits = c(0, 1), direction = 1) +
  facet_wrap(vars(station_code), labeller = labeller(station_code = label_wrap_gen())) +
  labs(x = "model", y = "image split") +
  theme_bw() +
  theme(
    aspect.ratio = 1,
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    strip.text.y = element_text(size = 6)
  )
ggsave("figs/tau-heat-station.png", plot = p, width = 8, height = 4)

p <- pred_tau %>%
  ggplot(aes(model, fct_rev(station_code))) +
  geom_tile(aes(fill = tau)) +
  geom_text(aes(label = sprintf("%.2f", tau)), size = 3) +
  scale_fill_distiller(palette = "YlGnBu", limits = c(0, 1), direction = 1) +
  facet_wrap(vars(split), ncol = 2) +
  labs(x = "model", y = "image split") +
  theme_bw() +
  theme(
    aspect.ratio = 1,
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1)
  )
ggsave("figs/tau-heat-split.png", plot = p, width = 10, height = 11)

pred_tau %>%
  group_by(station_code, split) %>%
  summarise(min = min(tau), max = max(tau), median = median(tau), .groups = "drop") %>%
  filter(split == "all") %>%
  arrange(desc(median))

p_z_ts <- pred %>%
  unnest(data) %>%
  mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out"))) %>%
  group_by(model, station_code) %>%
  mutate(across(c(value, prediction), scale)) %>%
  ungroup() %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value, linetype = "obs. flow")) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5) +
  scale_color_brewer(palette = "Set1") +
  facet_grid(vars(model), vars(station_code), scales = "free_x") +
  guides(
    linetype = guide_legend(order = 2),
    color = guide_legend(order = 1)
  ) +
  labs(
    x = "timestamp", y = "z(pred. score)",
    color = "image\nsplit", linetype = NULL
  ) +
  theme_bw()

p_z_splot <- pred %>%
  unnest(data) %>%
  mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out"))) %>%
  group_by(model, station_code) %>%
  mutate(across(c(value, prediction), scale)) %>%
  ungroup() %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  facet_grid(vars(model), vars(station_code), scales = "free_x", labeller = labeller(station_code = label_wrap_gen(width = 20))) +
  labs(
    x = "z(obs. flow)", y = "z(pred. score)",
    color = "image\nsplit"
  ) +
  theme_bw() +
  theme(
    axis.text.y = element_blank(),
    axis.title.y = element_blank()
    # aspect.ratio = 1
  )
p_z <- (p_z_ts | p_z_splot) +
  plot_layout(ncol = 2, widths = c(3, 1), guides = "collect")
ggsave("figs/ts-splot-z.png", plot = p_z, width = 14, height = 9)

p_r_ts <- pred %>%
  unnest(data) %>%
  mutate(
    split = factor(split, levels = c("train", "val", "test-in", "test-out"))
  ) %>%
  group_by(model, station_code) %>%
  mutate(
    across(c(value, prediction), ~ (rank(.) - 1) / (n() - 1))
  ) %>%
  ungroup() %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value, linetype = "obs. flow")) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5) +
  scale_color_brewer(palette = "Set1") +
  scale_y_continuous(labels = scales::percent) +
  facet_grid(vars(model), vars(station_code), scales = "free_x") +
  guides(
    linetype = guide_legend(order = 2),
    color = guide_legend(order = 1)
  ) +
  labs(
    x = "timestamp", y = "rank(pred. score)",
    color = "image\nsplit", linetype = NULL
  ) +
  theme_bw()
p_r_splot <- pred %>%
  unnest(data) %>%
  mutate(
    split = factor(split, levels = c("train", "val", "test-in", "test-out"))
  ) %>%
  group_by(model, station_code) %>%
  mutate(
    across(c(value, prediction), ~ (rank(.) - 1) / (n() - 1))
  ) %>%
  ungroup() %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent) +
  facet_grid(vars(model), vars(station_code), scales = "free_x", labeller = labeller(station_code = label_wrap_gen(width = 20))) +
  labs(
    x = "rank(obs. flow)", y = "rank(pred. score)",
    color = "image\nsplit"
  ) +
  theme_bw() +
  theme(
    axis.text.y = element_blank(),
    axis.title.y = element_blank()
    # aspect.ratio = 1
  )

p_r <- (p_r_ts | p_r_splot) +
  plot_layout(ncol = 2, widths = c(3, 1), guides = "collect")
ggsave("figs/ts-splot-r.png", plot = p_r, width = 14, height = 9)

p_29 <- wrap_plots(
  p_29_z_ts +
    ggtitle("(a) Normalized Values"),
  p_29_z_splot +
    labs(y = NULL) +
    theme(axis.text.y = element_blank(), axis.title.x = element_blank()),
  p_29_r_ts +
    ggtitle("(b) Rank Percentiles"),
  p_29_r_splot +
    labs(y = NULL) +
    theme(axis.text.y = element_blank(), axis.title.x = element_blank()),
  widths = c(3, 1, 3, 1),
  guides = "collect"
)
ggsave("figs/ts-splot-29.png", plot = p_29, width = 14, height = 9)


# splot: by model
p <- pred %>%
  unnest(data) %>%
  mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out"))) %>%
  group_by(model, station_code) %>%
  mutate(across(c(value, prediction), scale)) %>%
  ungroup() %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
  geom_blank(aes(prediction, value)) +
  scale_color_brewer(palette = "Set1") +
  facet_grid(vars(model), vars(station_code), labeller = labeller(station_code = label_wrap_gen())) +
  labs(
    x = "z(obs. flow)", y = "z(pred. score)",
    color = "image\nsplit"
  ) +
  theme_bw() +
  theme(
    strip.text.x = element_text(size = 6),
    aspect.ratio = 1
  )
ggsave("figs/splot-stations.png", plot = p, width = 14, height = 8)

## pdf: diagnostics --------------------------------------------------------

# pdf: ts+splot one station by model
pred_tau_gt <- pred_tau %>%
  select(-rmse) %>%
  nest_by(station_code) %>%
  mutate(
    t_tau = list({
      data %>%
        pivot_wider(names_from = "split", values_from = "tau") %>%
        gt(caption = "Kendall's tau") %>%
        cols_align(align = "right", columns = -model) %>%
        cols_align(align = "left", columns = model) %>%
        fmt_number(decimals = 3) %>%
        data_color(columns = -model, palette = "YlGnBu", domain = c(0, 1))
    })
  ) %>%
  ungroup() %>%
  select(-data)

p_pdf_z <- pred %>%
  ungroup() %>%
  nest_by(station_code, .keep = TRUE) %>%
  left_join(pred_tau_gt, by = "station_code") %>%
  mutate(
    p_ts = list({
      data %>%
        unnest(data) %>%
        mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out"))) %>%
        group_by(model, station_code) %>%
        mutate(across(c(value, prediction), scale)) %>%
        ungroup() %>%
        ggplot(aes(timestamp)) +
        geom_line(aes(y = value, linetype = "z(obs. flow)")) +
        geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5) +
        scale_color_brewer(palette = "Set1") +
        facet_wrap(vars(model), ncol = 1, strip.position = "right") +
        guides(
          linetype = guide_legend(order = 2),
          color = guide_legend(order = 1)
        ) +
        labs(
          x = "timestamp", y = "z(pred. score)",
          color = "image\nsplit", linetype = NULL
        ) +
        theme_bw() +
        theme(
          strip.text.y = element_text(size = 6)
        )
    }),
    p_splot = list({
      data %>%
        unnest(data) %>%
        mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out"))) %>%
        group_by(model, station_code) %>%
        mutate(across(c(value, prediction), scale)) %>%
        ungroup() %>%
        ggplot(aes(value, prediction)) +
        geom_abline() +
        geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
        geom_blank(aes(prediction, value)) +
        scale_color_brewer(palette = "Set1") +
        facet_wrap(vars(model), ncol = 1, strip.position = "right") +
        labs(
          x = "z(obs. flow)", y = "z(pred. score)",
          color = "image\nsplit"
        ) +
        theme_bw() +
        theme(
          strip.text.y = element_text(size = 6),
          aspect.ratio = 1
        )
    }),
    p = list({
      p <- (p_ts | p_splot) +
        plot_layout(ncol = 2, widths = c(4, 1), guides = "collect")
      (p / (wrap_table(t_tau) + ggtitle("Kendall's tau"))) +
        plot_layout(ncol = 1, heights = c(3, 1)) +
        plot_annotation(title = glue("{station_code} | Normalized Values"))
    })
  )
p_pdf_z$p[[1]]

p_pdf_r <- pred %>%
  ungroup() %>%
  nest_by(station_code, .keep = TRUE) %>%
  left_join(pred_tau_gt, by = "station_code") %>%
  mutate(
    p_ts = list({
      data %>%
        unnest(data) %>%
        mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out"))) %>%
        group_by(model, station_code) %>%
        mutate(across(c(value, prediction), ~ (rank(.) - 1) / (n() - 1))) %>%
        ungroup() %>%
        ggplot(aes(timestamp)) +
        geom_line(aes(y = value, linetype = "z(obs. flow)")) +
        geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5) +
        scale_color_brewer(palette = "Set1") +
        scale_y_continuous(labels = scales::percent) +
        facet_wrap(vars(model), ncol = 1, strip.position = "right") +
        guides(
          linetype = guide_legend(order = 2),
          color = guide_legend(order = 1)
        ) +
        labs(
          x = "timestamp", y = "rank(pred. score)",
          color = "image\nsplit", linetype = NULL
        ) +
        theme_bw() +
        theme(
          strip.text.y = element_text(size = 6)
        )
    }),
    p_splot = list({
      data %>%
        unnest(data) %>%
        mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out"))) %>%
        group_by(model, station_code) %>%
        mutate(across(c(value, prediction), ~ (rank(.) - 1) / (n() - 1))) %>%
        ungroup() %>%
        ggplot(aes(value, prediction)) +
        geom_abline() +
        geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
        geom_blank(aes(prediction, value)) +
        scale_color_brewer(palette = "Set1") +
        scale_x_continuous(labels = scales::percent) +
        scale_y_continuous(labels = scales::percent) +
        facet_wrap(vars(model), ncol = 1, strip.position = "right") +
        labs(
          x = "rank(obs. flow)", y = "rank(pred. score)",
          color = "image\nsplit"
        ) +
        theme_bw() +
        theme(
          strip.text.y = element_text(size = 6),
          aspect.ratio = 1
        )
    }),
    p = list({
      p <- (p_ts | p_splot) +
        plot_layout(ncol = 2, widths = c(4, 1), guides = "collect")
      (p / (wrap_table(t_tau) + ggtitle("Kendall's tau"))) +
        plot_layout(ncol = 1, heights = c(3, 1)) +
        plot_annotation(title = glue("{station_code} | Rankings"))
    })
  )
p_pdf_r$p[[1]]

pdf("figs/ts-splot.pdf", width = 14, height = 11)
for (i in 1:nrow(p_pdf_z)) {
  print(p_pdf_z$p[[i]])
  print(p_pdf_r$p[[i]])
}
dev.off()


## other figures -----------------------------------------------------------


# ts: by stations
pred %>%
  unnest(data) %>%
  mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out"))) %>%
  group_by(model, station_code) %>%
  mutate(across(c(value, prediction), scale)) %>%
  ungroup() %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = split), size = 0.5, alpha = 0.5) +
  scale_color_brewer(palette = "Set1") +
  scale_x_datetime(limits = c(ymd_hm(202101010000), NA)) +
  facet_grid(vars(station_code), vars(model), labeller = labeller(station_code = label_wrap_gen()), scales = "free") +
  labs(
    x = "timestamp", y = "z(pred. score)",
    color = "image\nsplit"
  ) +
  theme_bw() +
  theme(
    strip.text.y = element_text(size = 6)
  )

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
pred_tau %>%
  select(-rmse) %>%
  pivot_wider(names_from = "model", values_from = "tau")

# tau: dot plot by split, color=model
pred_tau %>%
  ggplot(aes(station_code, tau)) +
  geom_point(aes(color = model), size = 3, alpha = 0.75) +
  coord_flip() +
  scale_color_brewer(palette = "Set1") +
  facet_grid(vars(split)) +
  ylim(NA, 1) +
  labs(x = "station", y = "tau") +
  theme_bw()


# tau: dot plot by model, color=split
pred_tau %>%
  ggplot(aes(station_code, tau)) +
  geom_point(aes(color = split), size = 3, alpha = 0.75) +
  coord_flip() +
  scale_color_brewer(palette = "Set1") +
  facet_grid(vars(model)) +
  ylim(NA, 1) +
  labs(x = "station", y = "tau") +
  theme_bw()

# tau: dot plot by split, color=model
pred_tau %>%
  ggplot(aes(fct_rev(model), tau)) +
  geom_hline(yintercept = 0, alpha = 0.5) +
  geom_point(aes(color = split), size = 3, alpha = 0.75) +
  coord_flip() +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(station_code), labeller = labeller(station_code = label_wrap_gen())) +
  ylim(NA, 1) +
  labs(x = "model", y = "tau") +
  theme_bw() +
  theme(
    strip.text.y = element_text(size = 6)
  )

# tau: box plot by split, color=model
pred_tau %>%
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

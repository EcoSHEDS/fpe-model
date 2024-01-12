# export dataset for station and variable
# usage: Rscript dataset.R <station_id> <variable_id> </path/to/datasets>
# example: Rscript dataset.R 17 FLOW_CFS D:/fpe/datasets

Sys.setenv(TZ = "GMT")

library(tidyverse)
library(jsonlite)
library(lubridate)
library(logger)
library(janitor)
library(glue)

MAX_DTIME <- 65 # max value interp gap

args <- commandArgs(trailingOnly = TRUE)
station_id <- parse_number(args[1])
variable_id <- path.expand(args[2])
output_dir <- path.expand(args[3])

log_info("station_id: {station_id}")
log_info("variable_id: {variable_id}")
log_info("output_dir: {output_dir}")

# dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
stopifnot(dir.exists(output_dir))

config <- config::get()

con <- DBI::dbConnect(
  RPostgres::Postgres(),
  host = config$db$host,
  port = config$db$port,
  dbname = config$db$database,
  user = config$db$user,
  password = config$db$password
)

fetch_station <- function(con, station_id) {
  DBI::dbGetQuery(con, "select * from stations where id = $1", list(station_id)) %>%
    as_tibble()
}

fetch_station_values_from_fpe <- function(con, station_id, variable_id) {
  DBI::dbGetQuery(con, "
select st.name as station_name, d.station_id, s.dataset_id, s.id as series_id, s.variable_id, v.timestamp, v.value, v.flag from
datasets d
left join series s on d.id = s.dataset_id
left join values v on s.id = v.series_id
left join stations st on st.id = d.station_id
where d.station_id = $1
and d.status = 'DONE'
and s.variable_id = $2
", list(station_id, variable_id)) %>%
    as_tibble()
}

fetch_station_images <- function(con, station_id) {
  DBI::dbGetQuery(con, "
select s.name as station_name, iset.station_id, i.imageset_id, i.id as image_id, i.timestamp, i.filename, i.full_url as url from
imagesets iset
left join images i on iset.id = i.imageset_id
left join stations s on s.id = iset.station_id
where iset.station_id = $1
and i.status = 'DONE'
", list(station_id)) %>%
    as_tibble()
}

fetch_station_flows_from_nwis <- function(station, start, end) {
  x_raw <- dataRetrieval::readNWISuv(station$nwis_id, parameterCd = "00060", startDate = as_date(start) - days(1), endDate = as_date(end) + days(1)) %>%
    tibble()
  if (nrow(x_raw) == 0) return(tibble())
  dataRetrieval::renameNWISColumns(x_raw) %>%
    select(timestamp = dateTime, value = Flow_Inst, flag = Flow_Inst_cd) %>%
    mutate(
      station_name = station$name,
      station_id = station$id,
      dataset_id = "NWIS",
      series_id = NA_character_,
      variable_id = "FLOW_CFS",
      .before = "timestamp"
    ) %>%
    mutate(
      flag = na_if(flag, "A")
    ) %>%
    filter(!is.na(value))
}

log_info("fetching: station")
station <- fetch_station(con, station_id)
stopifnot(nrow(station) == 1)

log_info("station.name: {station$name[[1]]}")

data_dir <- file.path(output_dir, station$name[[1]], variable_id)
if (!dir.exists(data_dir)) {
  log_info("data_dir: {data_dir} (created)")
  dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
} else {
  log_info("data_dir: {data_dir} (exists)")
}

log_info("saving: {file.path(data_dir, 'station.json')}")
station %>%
  select(-metadata) %>%
  mutate(across(c(waterbody_type, status), as.character)) %>%
  as.list() %>%
  write_json(file.path(data_dir, "station.json"), auto_unbox = TRUE)

log_info("fetching: images")
images <- fetch_station_images(con, station_id)

log_info("fetching: values")
value_source <- if_else(!is.na(na_if(station$nwis_id, "")) && variable_id == "FLOW_CFS", "NWIS", "FPE")
if (value_source == "NWIS") {
  start_timestamp <- min(images$timestamp)
  end_timestamp <- max(images$timestamp)
  values <- fetch_station_flows_from_nwis(station, start_timestamp, end_timestamp)
} else {
  values <- fetch_station_values_from_fpe(con, station_id, variable_id)
}


# images -------------------------------------------------------------

find_max_dtime <- function(x, timestamps) {
  first_index_after_x <- which(timestamps > x)[1]
  if (is.na(first_index_after_x)) {
    # x > all timestamps
    return(NA_real_)
  } else if (first_index_after_x == 1) {
    # x < all timestamps
    return(NA_real_)
  }
  timestamp_before <- timestamps[first_index_after_x - 1]
  dtime_before <- as.numeric(difftime(timestamp_before, x, units = "mins"))
  timestamp_after <- timestamps[first_index_after_x]
  dtime_after <- as.numeric(difftime(timestamp_after, x, units = "mins"))
  max(dtime_before, dtime_after)
}

if (nrow(values) > 0) {
  obs_values <- values %>%
    filter(
      timestamp >= min(images$timestamp, na.rm = TRUE) - days(1),
      timestamp <= max(images$timestamp, na.rm = TRUE) + days(1),
      !is.na(value),
      floor_date(timestamp, unit = "months") != ymd(20220501)
    ) %>%
    group_by(timestamp) %>%
    summarise(value = mean(value))
    # mutate(
    #   timestamp_hr = floor_date(timestamp, unit = "hours")
    # ) %>%
    # complete(timestamp_hr = seq(min(timestamp_hr), max(timestamp_hr), by = "hour")) %>%
    # mutate(
    #   timestamp = coalesce(timestamp, timestamp_hr)
    # ) %>%
    # select(-timestamp_hr)

  interp_values <- approxfun(obs_values$timestamp, y = obs_values$value, na.rm = FALSE)
  value_timestamps <- sort(unique(obs_values$timestamp))

  log_info("estimating value for each image")
  images_values <- images %>%
    mutate(
      filename = map_chr(url, ~ httr::parse_url(.)$path),
      interp_value = interp_values(timestamp),
      max_dtime = map_dbl(timestamp, \(x) find_max_dtime(x, value_timestamps)),
      value = if_else(!is.na(max_dtime) & max_dtime <= MAX_DTIME, interp_value, NA_real_)
    ) %>%
    arrange(timestamp)

  p <- obs_values %>%
    ggplot(aes(timestamp, value)) +
    geom_line() +
    geom_point(
      data = filter(images_values, !is.na(value)),
      aes(y = coalesce(value, 0), color = "value"),
      size = 0.25
    ) +
    geom_point(
      data = filter(images_values, is.na(value)),
      aes(y = coalesce(value, 0), color = "no value"),
      size = 0.25
    ) +
    scale_color_manual(NULL, values = c("no value" = "orangered", "value" = "deepskyblue")) +
    labs(
      title = glue("{station$name[[1]]} (ID={station_id}) | {variable_id}")
    )
  log_info("saving: {file.path(data_dir, 'images.png')}")
  ggsave(file.path(data_dir, "images.png"), p, width = 8, height = 4)
} else {
  images_values <- images %>%
    mutate(
      filename = map_chr(url, ~ httr::parse_url(.)$path),
      interp_value = NA_real_,
      max_dtime = NA_real_,
      value = NA_real_
    ) %>%
    arrange(timestamp)
  p <- images_values %>%
    ggplot(aes(timestamp, coalesce(value, 0), color = "no value")) +
    geom_point(size = 0.25) +
    scale_color_manual(NULL, values = c("no value" = "orangered", "value" = "deepskyblue"), drop = FALSE) +
    labs(
      y = "value",
      title = glue("{station$name[[1]]} (ID={station_id}) | {variable_id}")
    )
  log_info("saving: {file.path(data_dir, 'images.png')}")
  ggsave(file.path(data_dir, "images.png"), p, width = 8, height = 4)
}

log_info("saving: {file.path(data_dir, 'images.csv')}")
images_values %>%
  write_csv(file.path(data_dir, "images.csv"), na = "")

# annotations -------------------------------------------------------------

log_info("fetching: annotations from db")
annotations_db <- tbl(con, "annotations") %>%
  filter(
    !flag,
    station_id == local(station_id)
  ) %>%
  left_join(
    select(tbl(con, "stations"), station_id = id, station_name = name),
    by = "station_id"
  ) %>%
  select(annotation_id = id, user_id, station_id, station_name, duration_sec, n, url) %>%
  collect()

log_info("fetching: annotations from s3")
annotations_raw <- annotations_db %>%
  filter(!is.na(url)) %>%
  rowwise() %>%
  mutate(
    data = list({
      url %>%
        read_json(simplifyVector = TRUE, flatten = TRUE) %>%
        as_tibble() %>%
        mutate(pair_id = row_number())
    })
  )

annotations <- annotations_raw %>%
  mutate(
    data = list({
      data %>%
        mutate(
          left.attributes = map_chr(left.attributes, \(x) str_c(x, collapse = ",")),
          right.attributes = map_chr(right.attributes, \(x) str_c(x, collapse = ","))
        ) %>%
        left_join(
          images_values %>%
            select(
              left.imageId = image_id,
              left.timestamp = timestamp,
              left.value = value,
              left.url = url,
              left.filename = filename
            ),
          by = "left.imageId"
        ) %>%
        left_join(
          images_values %>%
            select(
              right.imageId = image_id,
              right.timestamp = timestamp,
              right.value = value,
              right.url = url,
              right.filename = filename
            ),
          by = "right.imageId"
        ) %>%
        mutate(
          delta_value = abs(left.value - right.value),
          avg_value = (left.value + right.value) / 2,
          rel_delta_value = delta_value / avg_value,
          true_rank = case_when(
            left.value < right.value ~ "RIGHT",
            left.value > right.value ~ "LEFT",
            left.value == right.value ~ "SAME",
            TRUE ~ NA_character_
          )
        )
    })
  ) %>%
  unnest(data)

stopifnot(
  all(!is.na(annotations$left.url)),
  all(!is.na(annotations$right.url))
)

log_info("saving: {file.path(data_dir, 'annotations.csv')}")
annotations %>%
  write_csv(file.path(data_dir, "annotations.csv"), na = "")


# log -----------------------------------------------------------------

if (nrow(images) > 0) {
  images_log <- glue("
images:
  count:  {scales::comma(nrow(images_values))} ({scales::comma(nrow(filter(images_values, !is.na(value))))} with values)
  start:  {format(min(images$timestamp), usetz = TRUE)}
  end:    {format(max(images$timestamp), usetz = TRUE)}
")
} else {
  images_log <- glue("
images:
  count:  0
")
}

if (nrow(values) > 0) {
  values_log <- glue("
values:
  source: {value_source}
  count:  {scales::comma(nrow(values))}
  start:  {format(min(values$timestamp), usetz = TRUE)}
  end:    {format(max(values$timestamp), usetz = TRUE)}
  freq:   {median(as.numeric(difftime(sort(unique(values$timestamp)), lag(sort(unique(values$timestamp))), units = 'mins')), na.rm = TRUE)} min (median)
  maxgap: {MAX_DTIME} min
")
} else {
  values_log <- glue("
values:
  source: {value_source}
  count:  {scales::comma(nrow(values))}
")
}

if (nrow(annotations) > 0) {
  annotations_log <- glue("
annotations:
  count:  {scales::comma(nrow(annotations))}
  start:  {format(min(c(annotations$left.timestamp, annotations$right.timestamp)), usetz = TRUE)}
  end:    {format(max(c(annotations$left.timestamp, annotations$right.timestamp)), usetz = TRUE)}
")
} else {
  annotations_log <- glue("
annotations:
  count:  0
")
}

glue("
station:  {station$name[[1]]} (ID={station_id})
variable: {variable_id}
exported: {now()}
{images_log}
{values_log}
{annotations_log}
") %>%
  write_file(file.path(data_dir, "export.log"))

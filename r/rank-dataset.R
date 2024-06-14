# export rank dataset for given station and variable
# usage: Rscript rank-dataset.R --help
# example: Rscript rank-dataset.R -d /mnt/d/fpe/rank -s 29 -v FLOW_CFS -m 65 RANK-FLOW-20240326

Sys.setenv(TZ = "GMT")

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
  library(lubridate)
  library(logger)
  library(janitor)
  library(glue)
  library(httr2)
  library(optparse)
})

parser <- OptionParser()
parser <- add_option(
  parser, c("-d", "--directory"), type="character",
  help="Path to root directory"
)
parser <- add_option(
  parser, c("-s", "--station-id"), type="integer",
  help="Station ID from database"
)
parser <- add_option(
  parser, c("-v", "--variable-id"), type="character",
  default="FLOW_CFS", help="Variable ID from database (default='FLOW_CFS')"
)
parser <- add_option(
  parser, c("-m", "--maxgap"), type="integer", default=65,
  help="Maximum gap duration (minutes) for value interpolation (default=65)"
)
parser <- add_option(
  parser, c("-o", "--overwrite"), action="store_true",
  default=FALSE, help="Overwrite existing dataset (if exists)"
)

if (interactive()) {
  cmd_args <- c(
    "--directory=/mnt/d/fpe/rank",
    "--station-id=81",
    "--variable-id=FLOW_CFS",
    "--overwrite",
    "RANK-FLOW-20240613"
  )
} else {
  cmd_args <- commandArgs(trailingOnly = TRUE)
}

args <- parse_args(
  parser,
  positional_arguments = 1,
  convert_hyphens_to_underscores = TRUE,
  args = cmd_args
)

dataset_code <- args$args[1]
station_id <- args$options$station_id
variable_id <- args$options$variable_id
output_dir <- args$options$directory
overwrite <- args$options$overwrite
MAXGAP <- args$options$maxgap

log_info("dataset_code: {dataset_code}")
log_info("station_id: {station_id}")
log_info("variable_id: {variable_id}")
log_info("output_dir: {output_dir}")

# dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
stopifnot(dir.exists(output_dir))

dataset_dir <- file.path(output_dir, station_id, "datasets", dataset_code)
if (!dir.exists(dataset_dir)) {
  log_info("dataset_dir: {dataset_dir} (created)")
  dir.create(dataset_dir, showWarnings = FALSE, recursive = TRUE)
} else {
  if (overwrite) {
    log_warn("dataset_dir: {dataset_dir} (already exists, overwriting)")
  } else {
    log_error("dataset_dir: {dataset_dir} (already exists, use flag --overwrite to replace it)")
    stop("Dataset directory already exists")
  }
}

config <- config::get()

con <- DBI::dbConnect(
  RPostgres::Postgres(),
  host = config$db$host,
  port = config$db$port,
  dbname = config$db$database,
  user = config$db$user,
  password = config$db$password
)


# functions ---------------------------------------------------------------


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


# fetch -------------------------------------------------------------------

log_info("fetching: station")
station <- fetch_station(con, station_id)
stopifnot(nrow(station) == 1)
log_info("station: {station$name[[1]]}")

log_info("saving: {file.path(dataset_dir, 'station.json')}")
station %>%
  select(-metadata) %>%
  mutate(across(c(waterbody_type, status), as.character)) %>%
  as.list() %>%
  write_json(file.path(dataset_dir, "station.json"), auto_unbox = TRUE)

log_info("fetching: images")
images <- fetch_station_images(con, station_id)

value_source <- if_else(!is.na(na_if(station$nwis_id, "")) && variable_id == "FLOW_CFS", "NWIS", "FPE")
if (value_source == "NWIS") {
  log_info("fetching: values [NWIS]")
  start_timestamp <- min(images$timestamp)
  end_timestamp <- max(images$timestamp)
  values <- fetch_station_flows_from_nwis(station, start_timestamp, end_timestamp)
} else {
  log_info("fetching: values [FPE]")
  values <- fetch_station_values_from_fpe(con, station_id, variable_id)
}
log_info("values: {nrow(values)} rows")


# images -------------------------------------------------------------

if (nrow(values) > 0) {
  obs_values <- values %>%
    filter(
      timestamp >= min(images$timestamp, na.rm = TRUE) - days(1),
      timestamp <= max(images$timestamp, na.rm = TRUE) + days(1),
      !is.na(value)
    ) %>%
    group_by(timestamp) %>%
    summarise(value = mean(value))
} else {
  obs_values <- tibble()
}

if (nrow(obs_values) > 0) {
  interp_values <- approxfun(obs_values$timestamp, y = obs_values$value, na.rm = FALSE)
  value_timestamps <- sort(unique(obs_values$timestamp))

  log_info("interpolating values by image")
  images_values <- images %>%
    mutate(
      filename = map_chr(url, ~ httr::parse_url(.)$path),
      interp_value = interp_values(timestamp),
      max_dtime = map_dbl(timestamp, \(x) find_max_dtime(x, value_timestamps)),
      value = if_else(!is.na(max_dtime) & max_dtime <= MAXGAP, interp_value, NA_real_)
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
  log_info("saving: {file.path(dataset_dir, 'images.png')}")
  ggsave(file.path(dataset_dir, "images.png"), p, width = 8, height = 4)
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
  log_info("saving: {file.path(dataset_dir, 'images.png')}")
  ggsave(file.path(dataset_dir, "images.png"), p, width = 8, height = 4)
}

log_info("saving: {file.path(dataset_dir, 'images.csv')}")
images_values %>%
  select(-interp_value, -max_dtime) %>%
  write_csv(file.path(dataset_dir, "images.csv"), na = "")

if (nrow(values) > 0) {
  log_info("saving: {file.path(dataset_dir, 'values.csv')}")
  values %>%
    write_csv(file.path(dataset_dir, "values.csv"), na = "")

  p <- values %>%
    ggplot(aes(timestamp, value)) +
    geom_line() +
    labs(
      y = "value",
      title = glue("{station$name[[1]]} (ID={station_id}) | {variable_id}")
    )
  log_info("saving: {file.path(dataset_dir, 'values.png')}")
  ggsave(file.path(dataset_dir, "values.png"), p, width = 8, height = 4)
}

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
  mutate(
    data = map(url, function (url) {
      resp <- request(url) %>%
        req_retry(max_tries = 3) %>%
        req_perform()

      resp %>%
        resp_body_string() %>%
        fromJSON(simplifyVector = TRUE, flatten = TRUE) %>%
        as_tibble() %>%
        mutate(pair_id = row_number())
    }, .progress = TRUE)
  )

annotations <- annotations_raw %>%
  rowwise() %>%
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
  unnest(data) %>%
  filter(!is.na(left.url), !is.na(right.url))

stopifnot(
  all(!is.na(annotations$left.url)),
  all(!is.na(annotations$right.url))
)

if (nrow(annotations) > 0) {
  p <- annotations %>%
    ggplot(aes(left.timestamp, right.timestamp)) +
    geom_point(aes(color = rank), size = 0.5) +
    scale_x_datetime(date_breaks = "2 months", date_labels = "%b %Y") +
    scale_y_datetime(date_breaks = "2 months", date_labels = "%b %Y") +
    labs(
      title = glue("{station$name[[1]]} (ID={station_id}) | {variable_id}")
    ) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
  log_info("saving: {file.path(dataset_dir, 'annotations-splot.png')}")
  ggsave(file.path(dataset_dir, "annotations-splot.png"), p, width = 8, height = 6)

  p <- tibble(timestamp = c(annotations$left.timestamp, annotations$right.timestamp)) %>%
    arrange(timestamp) %>%
    mutate(n = row_number()) %>%
    ggplot(aes(timestamp, n)) +
    geom_line() +
    scale_x_datetime(date_breaks = "2 months", date_labels = "%b %Y") +
    labs(
      y = "cumul. # annotated images",
      title = glue("{station$name[[1]]} (ID={station_id}) | {variable_id}")
    ) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
  log_info("saving: {file.path(dataset_dir, 'annotations-cumul.png')}")
  ggsave(file.path(dataset_dir, "annotations-cumul.png"), p, width = 8, height = 4)
}

log_info("saving: {file.path(dataset_dir, 'annotations.csv')}")
annotations %>%
  write_csv(file.path(dataset_dir, "annotations.csv"), na = "")


# log -----------------------------------------------------------------

if (nrow(images) > 0) {
  out_images <- list(
    count = nrow(images),
    count_missing_value = sum(is.na(images_values$value)),
    start = format(with_tz(min(images$timestamp), tzone = station$timezone), usetz = TRUE),
    end = format(with_tz(max(images$timestamp), tzone = station$timezone), usetz = TRUE)
  )
} else {
  out_images <- list(count = 0)
}

if (nrow(values) > 0) {
  median_freq <- median(as.numeric(difftime(
    sort(unique(values$timestamp)),
    lag(sort(unique(values$timestamp))),
    units = 'mins'
  )), na.rm = TRUE)
  out_values <- list(
    source = value_source,
    count = nrow(values),
    start = format(with_tz(min(values$timestamp), tzone = station$timezone), usetz = TRUE),
    end = format(with_tz(max(values$timestamp), tzone = station$timezone), usetz = TRUE),
    freq = median_freq
  )
} else {
  out_values = list(
    count = 0
  )
}

if (nrow(annotations) > 0) {
  out_annotations <- list(
    count = nrow(annotations),
    start = format(with_tz(min(c(annotations$left.timestamp, annotations$right.timestamp)), tzone = station$timezone), usetz = TRUE),
    end = format(with_tz(max(c(annotations$left.timestamp, annotations$right.timestamp)), tzone = station$timezone), usetz = TRUE)
  )
} else {
  out_annotations <- list(count = 0)
}

list(
  dataset_code = dataset_code,
  station_id = station_id,
  variable_id = variable_id,
  images = out_images,
  values = out_values,
  annotations = out_annotations,
  args = args,
  created = format(now(tz = "US/Eastern"), usetz = TRUE)
) %>%
  write_json(file.path(dataset_dir, "rank-dataset.json"), pretty = TRUE, auto_unbox = TRUE)

DBI::dbDisconnect(con)

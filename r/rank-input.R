# generate model input from annotations
# usage: Rscript rank-input.R --help
# example: Rscript rank-input.R -d D:/fpe/rank -s 29 -v FLOW_CFS -V 20240326

Sys.setenv(TZ = "GMT")

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(lubridate)
  library(jsonlite)
  library(logger)
  library(glue)
  library(optparse)
})

default_version <- format(today(tz = "US/Eastern"), "%Y%m%d")
default_end <- format(today(tz = "US/Eastern"), "%Y-%m-%d")

parser <- OptionParser()
parser <- add_option(
  parser, c("-d", "--directory"), type="character",
  help="Path to root directory"
)
parser <- add_option(
  parser, c("-s", "--station"), type="integer",
  help="Station ID from database"
)
parser <- add_option(
  parser, c("-v", "--variable"), type="character",
  default="FLOW_CFS", help="Variable (default='FLOW_CFS')"
)
parser <- add_option(
  parser, c("-D", "--dataset-version"), type="character",
  default=default_version,
  help=glue("Dataset version (default='{default_version}')")
)
parser <- add_option(
  parser, c("-m", "--model-version"), type="character",
  default=default_version,
  help=glue("Model version (default='{default_version}')")
)
parser <- add_option(
  parser, c("-f", "--frac-train"), type="double", default=0.8,
  help=glue("Fraction of annotations in training split (default=0.8)")
)
parser <- add_option(
  parser, c("-S", "--seed"), type="integer", default=NULL,
  help="Random seed (default=NULL, no seed)"
)

parser <- add_option(
  parser, "--min-hour", type="integer", default=7,
  help="Minimum hour for images (local time zone of station) (default=7, aka 7 AM)"
)
parser <- add_option(
  parser, "--max-hour", type="integer", default=18,
  help="Maximum hour for images (local time zone of station) (default=18, aka 6 PM)"
)

parser <- add_option(
  parser, "--min-month", type="integer", default=1,
  help="Minimum month for images (default=1, aka January)"
)
parser <- add_option(
  parser, "--max-month", type="integer", default=12,
  help="Maximum month for images (default=12, aka December)"
)

parser <- add_option(
  parser, "--images-start", type="character", default='2000-01-01',
  help="Minimum date (ISO Format, YYYY-MM-DD) for images (default='2000-01-01')"
)
parser <- add_option(
  parser, "--images-end", type="character", default=default_end,
  help=glue("Maximum date (ISO Format, YYYY-MM-DD) for images (default='{default_end}')")
)

parser <- add_option(
  parser, "--annotations-start", type="character", default='2000-01-01',
  help="Minimum date (ISO Format, YYYY-MM-DD) for annotation pairs (default='2000-01-01')"
)
parser <- add_option(
  parser, "--annotations-end", type="character", default=default_end,
  help=glue("Maximum date (ISO Format, YYYY-MM-DD) for annotation pairs (default='{default_end}')")
)

parser <- add_option(
  parser, c("-o", "--overwrite"), action="store_true",
  default=FALSE, help="Overwrite existing input dataset (if exists)"
)

if (interactive()) {
  args <- parse_args(parser, args = c(
    "--directory=/mnt/d/fpe/rank",
    "--station=65",
    "--variable=FLOW_CFS",
    "--overwrite",
    "--dataset-version=20240327",
    "--model-version=20240328",
    "--annotations-end=2023-09-30"
  ))
} else {
  args <- parse_args(parser, args = commandArgs(trailingOnly = TRUE))
}

output_dir <- args$directory
station_id <- args$station
variable_id <- args$variable
dataset_version <- args$`dataset-version`
model_version <- args$`model-version`
frac_train <- args$`frac-train`
seed <- args$seed
images_hours <- c(args$`min-hour`, args$`max-hour`)
images_months <- c(args$`min-month`, args$`max-month`)
images_dates <- ymd(c(args$`images-start`, args$`images-end`))
annotations_dates <- ymd(c(args$`annotations-start`, args$`annotations-end`))
overwrite <- args$overwrite

config <- config::get()

stopifnot(dir.exists(output_dir))

# fetch: station ----------------------------------------------------------

log_info("output_dir: {output_dir}")
log_info("station_id: {station_id}")
log_info("variable_id: {variable_id}")
log_info("dataset_version: {dataset_version}")
log_info("model_version: {model_version}")
log_info("frac_train: {frac_train}")
if (is.null(seed)) {
  log_info("seed: null")
} else {
  log_info("seed: {seed}")
}

log_info("images_hours: {str_c(images_hours, collapse=', ')}")
log_info("images_months: {str_c(images_months, collapse=', ')}")
log_info("images_dates: {str_c(images_dates, collapse=', ')}")
log_info("annotations_dates: {str_c(annotations_dates, collapse=', ')}")

log_info("fetching: station")
con <- DBI::dbConnect(
  RPostgres::Postgres(),
  host = config$db$host,
  port = config$db$port,
  dbname = config$db$database,
  user = config$db$user,
  password = config$db$password
)

station <- DBI::dbGetQuery(con, "select * from stations where id = $1", list(station_id)) %>%
  as_tibble()

DBI::dbDisconnect(con)

stopifnot(nrow(station) == 1)


# setup -------------------------------------------------------------------

dataset_dir <- file.path(output_dir, station$name[[1]], variable_id, dataset_version)
log_info("dataset_dir: {dataset_dir}")
stopifnot(dir.exists(dataset_dir))

model_dir <- file.path(dataset_dir, "models", model_version)
if (dir.exists(model_dir)) {
  if (overwrite) {
    log_warn("model_dir: {model_dir} (exists, overwriting)")
    unlink(model_dir, recursive = TRUE)
    dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
  } else {
    log_error("model_dir: {model_dir} (exists, use flag --overwrite to replace it)")
    stop("Model directory already exists")
  }
} else {
  log_info("model_dir: {model_dir} (created)")
  dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
}

input_dir <- file.path(model_dir, "input")
dir.create(input_dir, showWarnings = FALSE, recursive = TRUE)
log_info("input_dir: {input_dir}")


# functions ---------------------------------------------------------------

split_pairs <- function (x, frac_train = 0.8, seed = NULL) {
  pairs <- x %>%
    transmute(
      pair = row_number(),
      image_id_1 = left.imageId,
      timestamp_1 = left.timestamp,
      filename_1 = left.filename,
      value_1 = left.value,
      image_id_2 = right.imageId,
      timestamp_2 = right.timestamp,
      filename_2 = right.filename,
      value_2 = right.value,
      label = case_when(
        rank == "SAME" ~ 0,
        rank == "LEFT" ~ 1,
        rank == "RIGHT" ~ -1
      )
    )

  if (!is.null(seed)) {
    log_info('setting seed ({seed})')
    set.seed(seed)
  }
  pairs_train <- pairs %>%
    slice_sample(prop = frac_train)
  pairs_val <- pairs %>%
    filter(!pair %in% pairs_train$pair)

  bind_rows(
    train = duplicate_pairs(pairs_train),
    val = duplicate_pairs(pairs_val),
    .id = "split"
  )
}

duplicate_pairs <- function (x) {
  x_dup <- bind_cols(
    select(x, pair),
    select(x, ends_with("_2")) %>%
      rename_with(~ str_replace(., "2", "1")),
    select(x, ends_with("_1")) %>%
      rename_with(~ str_replace(., "1", "2")),
    transmute(x, label = -1 * label)
  )
  bind_rows(x, x_dup) %>%
    arrange(pair)
}

export_input <- function(station, images, annotations, pairs, dir) {
  subtitle <- glue("{station$name[[1]]} (ID={station_id}) | {variable_id}\nDataset: {dataset_version} | Model: {model_version}")

  stopifnot(all(pairs$image_id_1 %in% images$image_id))
  stopifnot(all(pairs$image_id_2 %in% images$image_id))

  image_ids_train <- pairs %>%
    filter(split == "train") %>%
    select(image_id_1, image_id_2) %>%
    pivot_longer(everything()) %>%
    pull(value) %>%
    unique()
  image_ids_val <- pairs %>%
    filter(split == "val") %>%
    select(image_id_1, image_id_2) %>%
    pivot_longer(everything()) %>%
    pull(value) %>%
    unique()
  images_split <- images %>%
    mutate(
      split = case_when(
        image_id %in% c(image_ids_train) ~ "train",
        image_id %in% c(image_ids_val) ~ "val",
        as_date(timestamp) <= annotations_dates[2] & as_date(timestamp) >= annotations_dates[1] ~ "test-in",
        TRUE ~ "test-out"
      ),
      split = factor(split, levels = c("train", "val", "test-in", "test-out"))
    ) %>%
    relocate(split)

  images_split %>%
    select(split, image_id, timestamp, filename, url, value) %>%
    write_csv(file.path(dir, "images.csv"))

  p <- images_split %>%
    ggplot(aes(timestamp, fct_rev(split), color = split)) +
    geom_point(size = 2, alpha = 0.2) +
    scale_color_brewer(palette = "Set1", guide = "none") +
    scale_x_datetime(date_breaks = "2 months", date_labels = "%b %Y") +
    labs(
      y = "split",
      title = "Timeseries of Photo Splits",
      subtitle = subtitle
    ) +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
    )
  ggsave(file.path(dir, "images-ts.png"), plot = p, width = 8, height = 4)

  p <- images_split %>%
    count(split, month = month(timestamp, label = TRUE, abbr = TRUE)) %>%
    ggplot(aes(month, n, fill = split)) +
    geom_col(position = "dodge") +
    scale_fill_brewer(palette = "Set1", guide = "none") +
    facet_wrap(vars(split), ncol = 1, scales = "free_y") +
    labs(
      y = "# photos",
      title = "Monthly Photo Histograms",
      subtitle = subtitle
    )
  ggsave(file.path(dir, "images-months.png"), plot = p, width = 6, height = 8)

  p <- images_split %>%
    ggplot(aes(timestamp, coalesce(value, 0), color = split, shape = is.na(value))) +
    geom_point(size = 1, alpha = 0.5) +
    scale_color_brewer(palette = "Set1") +
    labs(
      y = NULL,
      title = "Timeseries of Photo Values",
      subtitle = subtitle
    )
  ggsave(file.path(dir, "images-values-ts.png"), plot = p, width = 8, height = 6)

  if (sum(!is.na(images_split$value)) > 0) {
    p <- images_split %>%
      ggplot(aes(pmax(value, 0.1), color = split)) +
      stat_ecdf() +
      scale_color_brewer(palette = "Set1") +
      scale_x_log10() +
      scale_y_continuous(labels = scales::percent, breaks = scales::pretty_breaks(n = 10)) +
      labs(
        y = NULL,
        title = "Cumulative Frequency Distribution of Photo Values",
        subtitle = subtitle
      )
    ggsave(file.path(dir, "images-values-cfd.png"), plot = p, width = 8, height = 6)

    p <- images_split %>%
      ggplot(aes(pmax(value, 0.1), fill = split)) +
      geom_histogram() +
      scale_fill_brewer(palette = "Set1", guide = "none") +
      scale_x_log10() +
      facet_wrap(vars(split), ncol = 1, scale = "free_y") +
      #scale_y_continuous(labels = scales::percent, breaks = scales::pretty_breaks(n = 10)) +
      labs(
        y = "# photos",
        title = "Histograms of Photo Values",
        subtitle = subtitle
      )
    ggsave(file.path(dir, "images-values-hist.png"), plot = p, width = 8, height = 6)
  }

  annotations %>%
    write_csv(file.path(dir, "annotations.csv"))

  pairs %>%
    write_csv(file.path(dir, "pairs.csv"))

  p <- pairs %>%
    ggplot(aes(timestamp_1, timestamp_2)) +
    geom_point(aes(color = factor(label)), size = 0.5) +
    geom_abline() +
    scale_color_brewer(palette = "Set1") +
    scale_x_datetime(date_breaks = "2 months", date_labels = "%b %Y") +
    scale_y_datetime(date_breaks = "2 months", date_labels = "%b %Y") +
    facet_wrap(vars(split), nrow = 1) +
    labs(
      title = "Annotated Pairs | Timestamps",
      color = "label",
      subtitle = subtitle
    ) +
    theme(
      aspect.ratio = 1,
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
    )
  ggsave(file.path(dir, "pairs-timestamps.png"), p, width = 8, height = 6)

  if (sum(!(is.na(pairs$value_1) & is.na(pairs$value_2))) > 0) {
    p <- pairs %>%
      ggplot(aes(pmax(value_1, 0.1), pmax(value_2, 0.1))) +
      geom_point(aes(color = factor(label)), size = 0.5) +
      geom_abline() +
      scale_color_brewer(palette = "Set1") +
      scale_x_log10(labels = scales::comma) +
      scale_y_log10(labels = scales::comma) +
      facet_wrap(vars(split), nrow = 1) +
      labs(
        title = "Annotated Pairs | Obs. Values",
        color = "label",
        subtitle = subtitle
      ) +
      theme(
        aspect.ratio = 1,
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
      )
    ggsave(file.path(dir, "pairs-values.png"), p, width = 8, height = 8)

    p1 <- pairs %>%
      ggplot(aes(pmax(abs(value_1 - value_2), 0.1))) +
      geom_histogram() +
      scale_x_log10(labels = scales::comma) +
      facet_wrap(vars(split), ncol = 1, scales = "free_y") +
      labs(
        title = "Annotated Pairs | Abs. Difference in Obs. Values",
        x = "abs delta = abs(value_1 - value_2)",
        y = "# annotations",
        subtitle = subtitle
      ) +
      theme(
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
      )

    p2 <- pairs %>%
      mutate(
        mean_value = (value_1 + value_2) / 2,
        abs_delta = abs(value_1 - value_2),
        rel_delta = abs_delta / mean_value
      ) %>%
      ggplot(aes(rel_delta)) +
      geom_histogram() +
      facet_wrap(vars(split), ncol = 1, scales = "free_y") +
      labs(
        title = "Annotated Pairs | Rel. Difference in Obs. Values",
        x = "rel delta = abs(value_1 - value_2) / mean(value_1, value_2)",
        y = "# annotations",
        subtitle = subtitle
      ) +
      theme(
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
      )
    p <- p1 / p2
    ggsave(file.path(dir, "pairs-values-delta.png"), p, width = 8, height = 8)
  }

  manifest <- pairs %>%
    select(filename_1, filename_2) %>%
    pivot_longer(everything()) %>%
    pull(value) %>%
    unique()

  manifest_json <- str_replace(
    toJSON(manifest, auto_unbox = TRUE),
    "\\[", "[{\"prefix\": \"s3://usgs-chs-conte-prod-fpe-storage/\"},"
  )
  write_file(manifest_json, file.path(input_dir, "manifest.json"))

  list(
    created_at = format_ISO8601(now(tz="US/Eastern"), usetz = TRUE),
    station = list(
      id = station$id,
      name = station$name
    ),
    images = list(
      n = nrow(images),
      start = format_ISO8601(min(images$timestamp), usetz = TRUE),
      end = format_ISO8601(max(images$timestamp), usetz = TRUE)
    ),
    pairs = list(
      source = "human",
      n = nrow(pairs) / 2,
      start = format_ISO8601(
        min(c(pairs$timestamp_1, pairs$timestamp_2)),
        usetz = TRUE
      ),
      end = format_ISO8601(
        max(c(pairs$timestamp_1, pairs$timestamp_2)),
        usetz = TRUE
      ),
      splits = list(
        train = sum(pairs$split == "train") / 2,
        val = sum(pairs$split == "val") / 2
      )
    ),
    args = args
  ) %>%
    write_json(file.path(dir, "rank-input.json"), auto_unbox = TRUE, pretty = TRUE)
}

# run ---------------------------------------------------------------------

log_info("loading: station.json")
station <- read_json(file.path(dataset_dir, "dataset", "station.json"))

log_info("loading: images.csv")
images_total <- read_csv(file.path(dataset_dir, "dataset", "images.csv"), show_col_types = FALSE) %>%
  mutate(
    timestamp = with_tz(timestamp, tzone = station$timezone)
  )
images <- images_total %>%
  filter(
    hour(timestamp) >= images_hours[1],
    hour(timestamp) <= images_hours[2],
    month(timestamp) >= images_months[1],
    month(timestamp) <= images_months[2],
    as_date(timestamp) >= images_dates[1],
    as_date(timestamp) <= images_dates[2]
  )
log_info("images: n={scales::comma(nrow(images))} ({scales::percent(nrow(images)/nrow(images_total), accuracy = 1)} of {scales::comma(nrow(images_total))} total)")

log_info("loading: annotations.csv")
annotations_total <- read_csv(file.path(dataset_dir, "dataset", "annotations.csv"), col_types = cols(comment = "c"), show_col_types = FALSE) %>%
  mutate(
    across(
      c(left.timestamp, right.timestamp),
      \(x) with_tz(x, tzone = station$timezone)
    )
  )

annotations <- annotations_total %>%
  filter(
    rank != "UNKNOWN",
    left.imageId %in% images$image_id,
    right.imageId %in% images$image_id,
    as_date(left.timestamp) >= annotations_dates[1],
    as_date(left.timestamp) <= annotations_dates[2],
    as_date(right.timestamp) >= annotations_dates[1],
    as_date(right.timestamp) <= annotations_dates[2]
  )
log_info("annotations: n={scales::comma(nrow(annotations))} ({scales::percent(nrow(annotations)/nrow(annotations_total), accuracy = 1)} of {scales::comma(nrow(annotations_total))} total)")

pairs <- split_pairs(annotations, frac_train, seed)

n_pairs_train <- sum(pairs$split == 'train') / 2
n_pairs_val <- sum(pairs$split == 'val') / 2
n_pairs_total <- n_pairs_train + n_pairs_val
log_info("pairs(train): n={scales::comma(n_pairs_train)} ({scales::percent(n_pairs_train/n_pairs_total, accuracy = 1)})")
log_info("pairs(val): n={scales::comma(n_pairs_val)} ({scales::percent(n_pairs_val/n_pairs_total, accuracy = 1)})")

log_info("exporting: {input_dir}")
export_input(station, images, annotations, pairs, dir = input_dir)


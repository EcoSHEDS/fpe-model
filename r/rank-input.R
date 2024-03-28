# generate model input from annotations
# usage: Rscript rank-input.R --help
# example: Rscript rank-input.R -d D:/fpe/rank -s 29 -v FLOW_CFS -V 20240326

Sys.setenv(TZ = "GMT")

suppressPackageStartupMessages({
  library(tidyverse)
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
  parser, c("-V", "--version"), type="character",
  default=default_version,
  help=glue("Dataset version (default='{default_version}')")
)
parser <- add_option(
  parser, c("-i", "--model-id"), type="character",
  default=default_version,
  help=glue("Model ID (default='{default_version}')")
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
  parser, "--min-hour", type="integer", default=0,
  help="Minimum hour for images (local time zone of station) (default=0, aka 12:00 AM)"
)
parser <- add_option(
  parser, "--max-hour", type="integer", default=23,
  help="Maximum hour for images (local time zone of station) (default=23, aka 11:00 PM)"
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
    "--directory=D:/fpe/rank",
    "--station=29",
    "--variable=FLOW_CFS",
    "--overwrite",
    "--version=20240327",
    "--id=20240327",
    "--min-hour=7",
    "--max-hour=18",
    "--annotations-end=2023-08-31"
  ))
} else {
  args <- parse_args(parser, args = commandArgs(trailingOnly = TRUE))
}

output_dir <- args$directory
station_id <- args$station
variable_id <- args$variable
dataset_version <- args$version
model_id <- args$`model-id`
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
log_info("model_id: {model_id}")
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

model_dir <- file.path(dataset_dir, "models", model_id)
if (!dir.exists(model_dir)) {
  log_info("model_dir: {model_dir} (created)")
  dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
} else {
  if (overwrite) {
    log_warn("model_dir: {model_dir} (already exists, overwriting)")
  } else {
    log_error("model_dir: {model_dir} (already exists, use flag --overwrite to replace it)")
    stop("Model directory already exists")
  }
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

  list(
    train = duplicate_pairs(pairs_train),
    val = duplicate_pairs(pairs_val)
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

export_input <- function(images, annotations, pairs, dir) {
  subtitle <- glue("{station$name[[1]]} (ID={station_id}) | {variable_id}\nDataset: {dataset_version} | Model: {model_id}")

  images %>%
    write_csv(file.path(dir, "images.csv"))

  images_class <- images %>%
    mutate(
      class = case_when(
        image_id %in% c(pairs$train$image_id_1, pairs$train$image_id_2) ~ "train",
        image_id %in% c(pairs$val$image_id_1, pairs$val$image_id_2) ~ "val",
        TRUE ~ "unannotated"
      ),
      class = factor(class, levels = c("train", "val", "unannotated"))
    )
  p <- images_class %>%
    ggplot(aes(timestamp, fct_rev(class), color = class)) +
    geom_point(size = 2, alpha = 0.2) +
    scale_color_brewer(palette = "Set1", guide = "none") +
    scale_x_datetime(date_breaks = "2 months", date_labels = "%b %Y") +
    labs(
      y = "class",
      title = "Timeseries of Photo Classes",
      subtitle = subtitle
    ) +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
    )
  ggsave(file.path(dir, "images-ts.png"), plot = p, width = 8, height = 4)

  p <- images_class %>%
    ggplot(aes(value, color = class)) +
    stat_ecdf() +
    scale_color_brewer(palette = "Set1") +
    scale_x_log10() +
    scale_y_continuous(labels = scales::percent, breaks = scales::pretty_breaks(n = 10)) +
    labs(
      y = NULL,
      title = "Cumulative Frequency Distribution of Photo Values",
      subtitle = subtitle
    )
  ggsave(file.path(dir, "images-cfd.png"), plot = p, width = 8, height = 6)

  p <- images_class %>%
    ggplot(aes(value, fill = class)) +
    geom_histogram() +
    scale_fill_brewer(palette = "Set1", guide = "none") +
    scale_x_log10() +
    facet_wrap(vars(class), ncol = 1, scale = "free_y") +
    #scale_y_continuous(labels = scales::percent, breaks = scales::pretty_breaks(n = 10)) +
    labs(
      y = "# photos",
      title = "Histograms of Photo Values",
      subtitle = subtitle
    )
  ggsave(file.path(dir, "images-hist.png"), plot = p, width = 8, height = 6)

  p <- images_class %>%
    count(class, month = month(timestamp, label = TRUE, abbr = TRUE)) %>%
    ggplot(aes(month, n, fill = class)) +
    geom_col(position = "dodge") +
    scale_fill_brewer(palette = "Set1", guide = "none") +
    facet_wrap(vars(class), ncol = 1, scales = "free_y") +
    labs(
      y = "# photos",
      title = "Monthly Photo Histograms",
      subtitle = subtitle
    )
  ggsave(file.path(dir, "images-months.png"), plot = p, width = 6, height = 8)

  annotations %>%
    write_csv(file.path(dir, "annotations.csv"))

  pairs$train %>%
    write_csv(file.path(dir, "pairs-train.csv"))
  pairs$val %>%
    write_csv(file.path(dir, "pairs-val.csv"))

  pairs_df <- bind_rows(pairs, .id = "split")
  p <- pairs_df %>%
    ggplot(aes(timestamp_1, timestamp_2)) +
    geom_point(aes(color = factor(label)), size = 0.5) +
    geom_abline() +
    scale_x_datetime(date_breaks = "2 months", date_labels = "%b %Y") +
    scale_y_datetime(date_breaks = "2 months", date_labels = "%b %Y") +
    facet_wrap(vars(split), nrow = 1) +
    labs(
      title = "Annotations | Timestamps",
      color = "label",
      subtitle = subtitle
    ) +
    theme(
      aspect.ratio = 1,
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
    )
  ggsave(file.path(dir, "annotations-timestamps.png"), p, width = 8, height = 6)

  p <- pairs_df %>%
    ggplot(aes(value_1, value_2)) +
    geom_point(aes(color = factor(label)), size = 0.5) +
    geom_abline() +
    scale_x_log10(labels = scales::comma) +
    scale_y_log10(labels = scales::comma) +
    facet_wrap(vars(split), nrow = 1) +
    labs(
      title = "Annotations | Obs. Values",
      color = "label",
      subtitle = subtitle
    ) +
    theme(
      aspect.ratio = 1,
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
    )
  ggsave(file.path(dir, "annotations-values.png"), p, width = 8, height = 8)

  p <- pairs_df %>%
    ggplot(aes(abs(value_1 - value_2))) +
    geom_histogram() +
    scale_x_log10(labels = scales::comma) +
    facet_wrap(vars(split), ncol = 1, scales = "free_y") +
    labs(
      title = "Annotations | Abs. Difference in Obs. Values",
      x = "abs delta = abs(value_1 - value_2)",
      y = "# annotations",
      subtitle = subtitle
    ) +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
    )
  ggsave(file.path(dir, "annotations-absdelta.png"), p, width = 8, height = 8)

  p <- pairs_df %>%
    mutate(
      mean_value = (value_1 + value_2) / 2,
      abs_delta = abs(value_1 - value_2),
      rel_delta = abs_delta / mean_value
    ) %>%
    ggplot(aes(rel_delta)) +
    geom_histogram() +
    facet_wrap(vars(split), ncol = 1, scales = "free_y") +
    labs(
      title = "Annotations | Rel. Difference in Obs. Values",
      x = "rel delta = abs(value_1 - value_2) / mean(value_1, value_2)",
      y = "# annotations",
      subtitle = subtitle
    ) +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
    )
  ggsave(file.path(dir, "annotations-reldelta.png"), p, width = 8, height = 8)

  manifest <- bind_rows(
    pairs$train,
    pairs$val
  ) %>%
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
    source = "human",
    images = list(
      n = nrow(images),
      start = format_ISO8601(min(images$timestamp), usetz = TRUE),
      end = format_ISO8601(max(images$timestamp), usetz = TRUE)
    ),
    annotations = list(
      n = nrow(annotations),
      start = format_ISO8601(
        min(c(annotations$left.timestamp, annotations$right.timestamp)),
        usetz = TRUE
      ),
      end = format_ISO8601(
        max(c(annotations$left.timestamp, annotations$right.timestamp)),
        usetz = TRUE
      ),
      pairs = list(
        train = nrow(pairs$train) / 2,
        val = nrow(pairs$val) / 2
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
annotations_total <- read_csv(file.path(dataset_dir, "dataset", "annotations.csv"), show_col_types = FALSE) %>%
  mutate(
    across(
      c(left.timestamp, right.timestamp),
      \(x) with_tz(x, tzone = station$timezone)
    )
  )

n_unknown <- sum(annotations_total$rank == "UNKNOWN")

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
log_info("annotations: n={scales::comma(nrow(annotations))} ({scales::percent(nrow(annotations)/nrow(annotations_total), accuracy = 1)} of {scales::comma(nrow(annotations_total))} total, {n_unknown} UNKNOWN)")

pairs <- split_pairs(annotations, frac_train, seed)
n_pairs_total <- nrow(pairs$train) + nrow(pairs$val)
log_info("pairs(train): n={scales::comma(nrow(pairs$train)/2)} ({scales::percent(nrow(pairs$train)/n_pairs_total, accuracy = 1)})")
log_info("pairs(val): n={scales::comma(nrow(pairs$val)/2)} ({scales::percent(nrow(pairs$val)/n_pairs_total, accuracy = 1)})")

log_info("exporting: {input_dir}")
export_input(images, annotations, pairs, dir = input_dir)

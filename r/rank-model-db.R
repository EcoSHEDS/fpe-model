# create database rows for model
# usage: Rscript rank-model-db.R --help
# example: Rscript rank-model-db.R -t RANK -v FLOW_CFS -c RANK-FLOW-20240613 -s https://usgs-chs-conte-prod-fpe-storage.s3.us-west-2.amazonaws.com/models /mt/d/fpe/rank/stations.txt

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
  library(uuid)
})

parser <- OptionParser()
parser <- add_option(
  parser, c("-t", "--model-type"), type="character",
  default="RANK", help="Model type from database (default='RANK')"
)
parser <- add_option(
  parser, c("-v", "--variable-id"), type="character",
  default="FLOW_CFS", help="Variable ID from database (default='FLOW_CFS')"
)
parser <- add_option(
  parser, c("-c", "--model-code"), type="character",
  help="Model code"
)
parser <- add_option(
  parser, c("-u", "--s3-url"), type="character",
  default="https://usgs-chs-conte-prod-fpe-storage.s3.us-west-2.amazonaws.com/models",
  help="URL root to S3 bucket"
)

if (interactive()) {
  cmd_args <- c(
    "--model-code=RANK-FLOW-TEST",
    "/mnt/d/fpe/rank/stations-20240613.txt"
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

stations_file <- args$args[1]
model_type <- args$options$model_type
variable_id <- args$options$variable_id
s3_url <- args$options$s3_url
model_code <- args$options$model_code

log_info("model_type: {model_type}")
log_info("variable_id: {variable_id}")
log_info("s3_url: {s3_url}")
log_info("model_code: {model_code}")
log_info("stations_file: {stations_file}")

df <- read_csv(stations_file, col_names = "station_id", col_types = "d") %>%
  mutate(
    model_type_id = model_type,
    variable_id = variable_id,
    code = model_code,
    uuid = map_chr(station_id, UUIDgenerate),
    default = TRUE,
    diagnostics_url = glue("{s3_url}/{uuid}/{code}.html"),
    predictions_url = glue("{s3_url}/{uuid}/predictions.csv"),
    status = "DONE"
  )

db_filename <- glue("{tools::file_path_sans_ext(stations_file)}-models-db.csv")
log_info("saving model-db file: {db_filename} (n={nrow(df)})")
df %>%
  write_csv(db_filename)

uuid_filename <- glue("{tools::file_path_sans_ext(stations_file)}-models-uuid.txt")
log_info("saving models-uuid file: {uuid_filename} (n={nrow(df)})")
df %>%
  select(station_id, uuid) %>%
  write_delim(uuid_filename, delim = " ", col_names = FALSE)

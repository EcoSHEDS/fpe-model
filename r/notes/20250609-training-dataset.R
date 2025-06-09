# recreate training dataset from annotation file

library(tidyverse)
library(jsonlite)
library(janitor)
library(bit64)


# database ----------------------------------------------------------------

config <- config::get()

con <- DBI::dbConnect(
  RPostgres::Postgres(),
  host = config$db$host,
  port = config$db$port,
  dbname = config$db$database,
  user = config$db$user,
  password = config$db$password
)

images <- tbl(con, "images")



# load annotations --------------------------------------------------------

annotations <- read_json("~/tmp/annotations.json")


# pairs -------------------------------------------------------------------

x <- annotations[[1]]

convert_image <- function (x, i) {
  list(
    image_id = x$id,
    filename = x$filename,
    timestamp = lubridate::format_ISO8601(x$timestamp, usetz = TRUE),
    thumb_url = x$thumb_url,
    i = i
  )
}

convert_pair <- function (x, i) {
  left.image <- images %>%
    filter(id == x$left$imageId) %>%
    collect()
  right.image <- images %>%
    filter(id == x$right$imageId) %>%
    collect()
  list(
    left = convert_image(left.image, i),
    right = convert_image(right.image, i)
  )
}


toJSON(convert_pair(annotations[[1]], 1), auto_unbox = TRUE, pretty = TRUE)


pairs <- list()
for (i in 1:length(annotations)) {
  pairs[[i]] <- convert_pair(annotations[[i]], i)
}

pairs[[1]]
pairs[[2]]

write_json(list(station_id=15, pairs=pairs), "~/tmp/training.json", auto_unbox = TRUE)

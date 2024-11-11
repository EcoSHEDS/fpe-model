library(tidyverse)
library(glue)
library(ggrepel)
library(janitor)
library(patchwork)

base_dir <- "/mnt/d/fpe/rank"
output_dir <- "notes/20240613"
model_code <- "RANK-FLOW-20240613"

stations_ids <- read_csv("/mnt/d/fpe/rank/stations-20240613.txt", col_names = "id") %>%
  pull(id) %>%
  as.numeric()

config <- config::get()

con <- DBI::dbConnect(
  RPostgres::Postgres(),
  host = config$db$host,
  port = config$db$port,
  dbname = config$db$database,
  user = config$db$user,
  password = config$db$password
)

db_stations <- DBI::dbGetQuery(con, "select * from stations")

DBI::dbDisconnect(con)

stations <- db_stations %>%
  filter(id %in% stations_ids)

for (i in 1:nrow(stations)) {
  station <- stations[i, ]

  cat("copying:", station$id, "\n")

  src_dir <- glue("{base_dir}/{station$id}/models/{model_code}/")
  dest_dir <- glue("{output_dir}/{station$id}-{station$name}")
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  stopifnot(dir.exists(src_dir), dir.exists(dest_dir))

  file.copy(src_dir, dest_dir, overwrite = TRUE, recursive = TRUE)
}


stations %>%
  write_csv(glue("{output_dir}/stations.csv"))
base_dir <- "C:\Users\jdwalker\OneDrive - DOI\projects\fpe\Ranking Model\Model Results\2024-06-13 - USGS Stations"

stations <- read_csv(file.path(base_dir, "stations.csv"))
x <- stations %>%
  select(id, name) %>%
  rowwise() %>%
  mutate(
    pred = list({
      read_csv(glue("{base_dir}/{id}/models/{model_code}/transform/predictions.csv"))
    }),
    data = list({
      read_csv(glue("{base_dir}/{id}/models/{model_code}/input/annotations.csv"), col_types = cols(.default = col_character())) %>%
        select(left.imageId, left.attributes, right.imageId, right.attributes) %>%
        mutate(row = row_number()) %>%
        pivot_longer(-row, names_sep = "\\.", names_to = c("side", "name")) %>%
        pivot_wider() %>%
        transmute(image_id = as.numeric(imageId), label = attributes) %>%
        separate_longer_delim(label, delim = ",") %>%
        left_join(
          select(pred, image_id, timestamp, score),
          by = "image_id"
        )
    })
  ) %>%
  select(-pred)

x %>%
  unnest(data) %>%
  tabyl(label)

x %>%
  unnest(data) %>%
  mutate(
    label = factor(
      coalesce(label, "NORMAL"),
      levels = c("NORMAL", "DISCONNECTED", "DRY", "ICE_PARTIAL", "ICE", "BAD")
    )
  ) %>%
  ggplot(aes(label, score)) +
  geom_boxplot() +
  facet_wrap(vars(id, name))



x %>%
  unnest(data) %>%
  mutate(
    label = factor(
      coalesce(label, "NORMAL"),
      levels = c("NORMAL", "DISCONNECTED", "DRY", "ICE_PARTIAL", "ICE", "BAD")
    )
  ) %>%
  filter(
    str_sub(name, 1, 2) %in% c("SR", "PA", "PI"),
    label %in% c("NORMAL", "DISCONNECTED", "DRY")
  ) %>%
  ggplot(aes(label, score)) +
  geom_boxplot() +
  facet_wrap(vars(id, name))

x %>%
  unnest(data) %>%
  mutate(
    label = factor(
      coalesce(label, "NORMAL"),
      levels = c("NORMAL", "DISCONNECTED", "DRY", "ICE_PARTIAL", "ICE", "BAD")
    )
  ) %>%
  filter(
    str_sub(name, 1, 2) %in% c("PA", "PI", "SR"),
    label %in% c("NORMAL", "DISCONNECTED", "DRY")
  ) %>%
  group_by(id) %>%
  mutate(rank = rank(score) / n()) %>%
  ungroup() %>%
  ggplot(aes(rank, score)) +
  geom_line() +
  geom_point(
    data = ~ filter(., label %in% c("DISCONNECTED", "DRY")),
    aes(color = label),
    size = 1, alpha = 0.5
  ) +
  scale_x_continuous(labels = scales::percent) +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(name), scale = "free_x") +
  theme_bw() +
  theme(aspect.ratio = 1)

x %>%
  unnest(data) %>%
  mutate(
    label = factor(
      coalesce(label, "NORMAL"),
      levels = c("NORMAL", "DISCONNECTED", "DRY", "ICE_PARTIAL", "ICE", "BAD")
    )
  ) %>%
  filter(
    str_sub(name, 1, 2) %in% c("PA", "PI", "SR"),
    label %in% c("NORMAL", "DISCONNECTED", "DRY")
  ) %>%
  ggplot(aes(label, score)) +
  geom_line() +
  geom_jitter(
    data = ~ filter(., label %in% c("DISCONNECTED", "DRY")),
    aes(color = label),
    size = 1, alpha = 0.5, height = 0
  ) +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(name)) +
  theme_bw() +
  theme(aspect.ratio = 1)


x %>%
  unnest(data) %>%
  mutate(
    label = factor(
      coalesce(label, "NORMAL"),
      levels = c("NORMAL", "DISCONNECTED", "DRY", "ICE_PARTIAL", "ICE", "BAD")
    )
  ) %>%
  filter(
    str_sub(name, 1, 2) %in% c("PA", "PI", "SR"),
    label %in% c("NORMAL", "DISCONNECTED", "DRY")
  ) %>%
  ggplot(aes(timestamp, score)) +
  geom_line() +
  geom_point(
    data = ~ filter(., label %in% c("DISCONNECTED", "DRY")),
    aes(color = label),
    size = 1, alpha = 0.5
  ) +
  scale_x_datetime(date_breaks = "2 months", date_labels = "%b %Y") +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(name), ncol = 1) +
  theme_bw()

# generate dataset for 20250107-multisite

library(tidyverse)
library(jsonlite)
library(janitor)
library(glue)

exp_dir <- "~/git/fpe-model/experiments/20250107-multisite/"


# fetch stations ----------------------------------------------------------
# list all stations from API

url <- "https://drekttvuk1.execute-api.us-west-2.amazonaws.com/api/public/stations"
api_stations <- read_json(url, flatten = TRUE, simplifyVector = TRUE) %>%
  as_tibble() %>%
  mutate(
    n_variables = map_int(variables, length),
    has_flow = map_lgl(variables, \(x) "FLOW_CFS" %in% x),
    has_data = (!is.na(nwis_id) & nwis_id != "") | (has_flow)
  ) %>%
  print()

api_stations %>%
  tabyl(has_data)

stations <- api_stations %>%
  filter(
    has_data,
    waterbody_type == "ST",
    str_detect(affiliation_code, "USGS")
  ) %>%
  select(
    affiliation_code, id, name, description, timezone, nwis_id, waterbody_type,
    images.start_date, images.end_date, images.count,
    variables
  )

view(stations)
stations %>%
  tabyl(affiliation_code)

stations %>%
  pull(id) %>%
  write_lines(file.path(exp_dir, "stations", "stations.txt"))

# then run:
# ./batch-station-dataset.sh /mnt/d/fpe/experiments/20250107-multisite/stations/stations.txt /mnt/d/fpe/experiments/20250107-multisite/stations FLOW_CFS


# load all images ---------------------------------------------------------

images <- stations %>%
  rowwise() %>%
  mutate(
    data = list({
      f <- file.path(exp_dir, "stations", id, "data", "images.csv")
      read_csv(f, show_col_types = FALSE) %>%
        mutate(
          timestamp = with_tz(timestamp, tz = timezone),
          value = log10(pmax(value, 0.001))
        ) %>%
        filter(!is.na(value))
    }),
    n_images = nrow(data)
  ) %>%
  filter(n_images > 1000)


# run 01 ------------------------------------------------------------------
# test: last 1 months (n=100)
# train/val: all but last 2 months (n=120)

set.seed(2102)

test_station_ids <- sample(images$id, size = 10, replace = FALSE)
images_01 <- images %>%
  select(affiliation_code, id, name, data) %>%
  mutate(
    site_split = if_else(id %in% test_station_ids, "test", "train")
  ) %>%
  mutate(
    data = list({
      if (site_split == "test") {
        out <- data %>%
          mutate(split = "test-out") %>%
          slice_sample(n = 650, replace = FALSE)
      } else {
        end <- max(data$timestamp)
        test_start <- floor_date(end - months(1), "day")
        test_out <- data %>%
          filter(timestamp >= test_start) %>%
          slice_sample(n = 200, replace = FALSE)
        train <- data %>%
          filter(timestamp < test_start) %>%
          slice_sample(n = 200, replace = FALSE)
        val <- data %>%
          filter(
            timestamp < test_start,
            !image_id %in% train$image_id
          ) %>%
          slice_sample(n = 50, replace = FALSE)
        test_in <- data %>%
          filter(
            timestamp < test_start,
            !image_id %in% train$image_id,
            !image_id %in% val$image_id
          ) %>%
          slice_sample(n = 200, replace = FALSE)
        out <- bind_rows(
          train = train,
          val = val,
          `test-in` = test_in,
          `test-out` = test_out,
          .id = "split"
        )
      }
      out
    })
  ) %>%
  unnest(data)

images_01 %>%
  tabyl(split, site_split) %>%
  adorn_totals(where = "both")

summary(images_01)

images_01 %>%
  ggplot(aes(timestamp, value)) +
  geom_point(aes(color = split), size = 0.5) +
  facet_wrap(vars(name), scales = "free")

images_01 %>%
  ggplot(aes(value)) +
  stat_ecdf(aes(color = split)) +
  facet_wrap(vars(name), scales = "free")

images_01 %>%
  ggplot(aes(value)) +
  stat_ecdf(aes(color = split))

dir.create(file.path(exp_dir, "runs", "01", "input"), recursive = TRUE)
images_01 %>%
  write_csv(file.path(exp_dir, "runs", "01", "input", "images.csv"))

images_01 %>%
  pull(filename) %>%
  unique() %>%
  toJSON(auto_unbox = TRUE) %>%
  str_replace(
    "\\[", "[{\"prefix\": \"s3://usgs-chs-conte-prod-fpe-storage/\"},"
  ) %>%
  write_file(file.path(exp_dir, "runs", "01", "input", "manifest.json"))

metrics_01 <- read_csv(file.path(exp_dir, "runs", "01", "output", "data", "metrics.csv"))

metrics_01 %>%
  ggplot(aes(epoch)) +
  geom_line(aes(y = train_loss, color = "train")) +
  geom_line(aes(y = val_loss, color = "val"))

pred_01 <- read_csv(file.path(exp_dir, "runs", "01", "output", "data", "predictions.csv")) %>%
  arrange(desc(site_split), station_name, timestamp) %>%
  mutate(
    station_name = fct_inorder(station_name),
    site_split = factor(site_split, levels = c("train", "test")),
    split = factor(split, levels = c("train", "val", "test-in", "test-out"))
  )

pred_01 %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_wrap(vars(station_name)) +
  theme(aspect.ratio = 1)

pred_01 %>%
  ggplot(aes(value, prediction)) +
  geom_abline() +
  geom_point(aes(color = split), size = 0.5) +
  geom_blank(aes(prediction, value)) +
  facet_grid(vars(site_split), vars(split)) +
  theme(aspect.ratio = 1)

pred_01 %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value)) +
  geom_point(aes(y = prediction, color = split), size = 0.5) +
  facet_wrap(vars(station_name), scales = "free")

pred_01 %>%
  group_by(site_split, split) %>%
  summarise(
    tau = cor(value, prediction, method = "kendall")
  ) %>%
  pivot_wider(names_from = "site_split", values_from = "tau")

pred_01 %>%
  group_by(site_split, station_name, split) %>%
  summarise(
    tau = cor(value, prediction, method = "kendall")
  ) %>%
  ggplot(aes(split, tau)) +
  geom_boxplot() +
  geom_jitter(aes(color = split), height = 0, width = 0.2, alpha = 0.5) +
  facet_wrap(vars(site_split), labeller = label_both)



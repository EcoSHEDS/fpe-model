# set up model inputs for Westbrook stations

Sys.setenv(TZ = "US/Eastern")

library(tidyverse)
library(lubridate)
library(jsonlite)
library(glue)

FPE_DIR <- "/mnt/d/fpe/rank"
VARIABLE <- "FLOW_CFS"
DATASET_VERSION <- "20240327"
MODEL_VERSION <- "20240328"

subtitle <- glue("Dataset: {DATASET_VERSION} | Model: {MODEL_VERSION}")

x <- tibble(
  dir = list.dirs(FPE_DIR, recursive = FALSE)
) %>%
  mutate(
    station_json = map(dir, function (x) {
      read_json(file.path(x, VARIABLE, DATASET_VERSION, "dataset", "station.json"))
    }),
    station_id = map_chr(station_json, \(x) x$id),
    station_name = map_chr(station_json, \(x) x$name),
    annotations = map(dir, function (x) {
      read_csv(file.path(x, VARIABLE, DATASET_VERSION, "dataset", "annotations.csv"), show_col_types = FALSE)
    }),
    images = map(dir, function (x) {
      read_csv(file.path(x, VARIABLE, DATASET_VERSION, "dataset", "images.csv"), show_col_types = FALSE)
    }),
  )

cutoffs <- tribble(
  ~station_id, ~station_name,              ~cutoff_date,
  9,   "Sanderson Brook_01171010",         "2023-09-30",
  10,  "West Brook Lower_01171090",        "2023-09-30",
  11,  "West Brook Upper_01171030",        "2022-12-31",
  12,  "Avery Brook_Bridge_01171000",      "2023-09-30",
  13,  "Avery Brook_Side_01171000",        "2023-09-30",
  14,  "Avery Brook_River Right_01171000", "2023-09-30",
  15,  "Avery Brook_River Left_01171000",  "2023-09-30",
  16,  "West Brook Reservoir_01171020",    "2023-09-30",
  17,  "West Whately_01171005",            "2023-09-30",
  18,  "Obear Brook Lower_01171070",       "2023-09-30",
  29,  "West Brook 0_01171100",            "2023-09-30",
  33,  "Mitchell Brook_01171080",          "2023-09-30",
  65,  "Green River_01170100",             "2023-09-30",
  68,  "West Branch Swift River_01174565", "2023-10-31",
  145, "West Brook Upper_New23_01171030",  "2023-10-31"
) %>%
  mutate(cutoff_date = ymd(cutoff_date))

annotations <- x %>%
  select(annotations) %>%
  unnest(annotations) %>%
  mutate(
    max_date = pmax(as_date(left.timestamp), as_date(right.timestamp))
  ) %>%
  left_join(cutoffs, by = c("station_id", "station_name"))

annotations %>%
  group_by(station_id, station_name, cutoff_date) %>%
  summarise(
    n_before = sum(max_date <= cutoff_date),
    n_after = sum(max_date > cutoff_date),
    p_after = n_after / (n_before + n_after),
    max_date = max(max_date),
  )

annotations %>%
  mutate(date = pmax(as_date(left.timestamp), as_date(right.timestamp))) %>%
  group_by(station_id, station_name) %>%
  summarise(
    end_date = as_date(quantile(as.numeric(date), probs = 0.95))
  )

annotations %>%
  count(user_id)

annotations %>%
  count(station_name)

annotations %>%
  ggplot(aes(left.timestamp, right.timestamp)) +
  geom_point(aes(color = rank), alpha = 0.5) +
  geom_hline(data = cutoffs, aes(yintercept = as.POSIXct(cutoff_date))) +
  geom_vline(data = cutoffs, aes(xintercept = as.POSIXct(cutoff_date))) +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(station_id, station_name, cutoff_date), nrow = 3) +
  labs(
    title = "Annotation Timestamps",
    subtitle = subtitle
  ) +
  theme_bw() +
  theme(aspect.ratio = 1)
ggsave("notes/20240328/annotations-timestamps.png", width = 12, height = 8, scale = 2)

annotations %>%
  mutate(
    class = case_when(
      max_date <= cutoff_date ~ "Before",
      TRUE ~ "After"
    )
  ) %>%
  count(station_id, station_name, class) %>%
  group_by(station_id, station_name) %>%
  mutate(
    p = n / sum(n)
  ) %>%
  pivot_longer(c(n, p)) %>%
  ggplot(aes(fct_rev(fct_inorder(str_c(station_id, station_name, sep = " - "))), value)) +
  geom_col(aes(fill = class), position = "stack") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
  scale_fill_brewer(NULL, palette = "Set1") +
  facet_wrap(vars(name), scales = "free_x", strip.position = "bottom", labeller = labeller(name = c(
    "n" = "# Annotations",
    "p" = "Frac. Annotations"
  ))) +
  coord_flip() +
  labs(
    x = NULL,
    y = NULL,
    title = "# Annotations Before/After Training Cutoff",
    subtitle = subtitle
  ) +
  theme_bw() +
  theme(
    strip.placement = "outside",
    strip.background = element_blank(),
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
  )
ggsave("notes/20240328/annotations-counts.png", width = 12, height = 12)

annotations %>%
  ggplot(aes(pmax(left.value, 1), pmax(right.value, 1))) +
  geom_point(aes(color = rank), alpha = 0.5) +
  scale_x_log10() +
  scale_y_log10() +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(station_id, station_name), nrow = 3) +
  labs(
    title = "Annotation Values",
    subtitle = subtitle
  ) +
  theme_bw()
ggsave("notes/20240328/annotations-values.png", width = 12, height = 8, scale = 2)

images <- x %>%
  select(images) %>%
  unnest(images) %>%
  left_join(cutoffs, by = c("station_id", "station_name"))

images %>%
  distinct(station_id, station_name, cutoff_date, date = as_date(timestamp)) %>%
  ggplot(aes(date, fct_rev(fct_inorder(str_c(station_id, station_name, sep = " - "))))) +
  geom_point(aes(color = date > cutoff_date), alpha = 0.5, size = 2) +
  scale_color_brewer(NULL, palette = "Set1", labels = c(
    "TRUE" = "After",
    "FALSE" = "Before"
  )) +
  labs(
    x = NULL,
    y = NULL,
    title = "Photos Before/After Training Cutoff",
    subtitle = subtitle
  ) +
  theme_bw()
# ggsave("notes/20240328/images-ts.png", width = 12, height = 12)


annotations_train <- annotations %>%
  filter(max_date <= cutoff_date)

images_classes <- images %>%
  mutate(
    class = case_when(
      image_id %in% c(annotations_train$left.imageId, annotations_train$right.imageId) ~ "Annotated",
      as_date(timestamp) <= cutoff_date ~ "Unannotated",
      TRUE ~ "Test"
    ),
    class = factor(class, levels = c("Annotated", "Unannotated", "Test"))
  )

images_classes %>%
  distinct(station_id, station_name, cutoff_date, class, date = as_date(timestamp)) %>%
  ggplot(aes(date, fct_rev(class))) +
  geom_point(aes(color = class), alpha = 0.5, size = 2) +
  scale_color_brewer(NULL, palette = "Set1") +
  facet_wrap(vars(station_id, station_name, cutoff_date)) +
  labs(
    x = NULL,
    y = NULL,
    title = "Photo Timeseries",
    subtitle = str_c(subtitle, " | Max Annotation: 2023-09-30")
  ) +
  theme_bw()
ggsave("notes/20240328/images-ts.png", width = 12, height = 12)

images_classes %>%
  count(station_id, station_name, class) %>%
  group_by(station_id, station_name) %>%
  mutate(p = n / sum(n)) %>%
  pivot_longer(c(n, p)) %>%
  # arrange(desc(class)) %>%
  ggplot(aes(fct_rev(fct_inorder(str_c(station_id, station_name, sep = " - "))), value)) +
  geom_col(aes(fill = class), position = "stack") +
  scale_fill_brewer(NULL, palette = "Set1") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
  coord_flip() +
  facet_wrap(vars(name), scales = "free_x", strip.position = "bottom", labeller = labeller(name = c(
    "n" = "# Annotations",
    "p" = "Frac. Annotations"
  ))) +
  labs(
    x = NULL,
    y = NULL,
    title = "Photo Annotation Classes",
    subtitle = subtitle
  ) +
  theme_bw() +
  theme(
    strip.placement = "outside",
    strip.background = element_blank(),
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
  )
ggsave("notes/20240328/images-counts.png", width = 12, height = 12)

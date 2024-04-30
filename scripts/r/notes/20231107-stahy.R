library(tidyverse)
library(lubridate)
library(glue)
library(patchwork)

params <- list(
  root_dir = "~/Dropbox/Work/ecosheds/fpe/rank",
  site = "WESTB0",
  model_id = "20230922"
)

site_dir <- file.path(params$root_dir, params$site)
model_dir <- file.path(site_dir, "models", params$model_id)

split_args_file <- file.path(model_dir, "input", "args.json")
metrics_file <- file.path(model_dir, "output", "metrics.csv")
pred_file <- file.path(model_dir, "transform", "predictions.csv")
station_file <- file.path(site_dir, "data", "station.json")
images_file <- file.path(site_dir, "data", "images.csv")
train_args_file <- file.path(model_dir, "output", "args.json")
train_file <- file.path(model_dir, "input", "pairs-train.csv")
val_file <- file.path(model_dir, "input", "pairs-val.csv")
test_file <- file.path(model_dir, "input", "pairs-test.csv")

split_labels <- c(
  "train" = "Train",
  "val" = "Val",
  "test" = "Test",
  "na" = "Not Annotated"
)

stn <- jsonlite::read_json(station_file)
split_args <- jsonlite::read_json(split_args_file)
if (split_args$method == "oracle") {
  split_args$num_val_pairs <- split_args$num_eval_pairs
  split_args$num_test_pairs <- split_args$num_eval_pairs
  split_args$notes <- "N/A"
}
train_args <- jsonlite::read_json(train_args_file)

metrics <- read_csv(metrics_file, show_col_types = FALSE)

images <- read_csv(images_file, show_col_types = FALSE)

pairs_train <- read_csv(train_file, show_col_types = FALSE)
pairs_val <- read_csv(val_file, show_col_types = FALSE)
pairs_test <- read_csv(test_file, show_col_types = FALSE)

pairs <- bind_rows(
  train = pairs_train,
  val = pairs_val,
  test = pairs_test,
  .id = "split"
)
pairs <- bind_rows(
  `1` = select(pairs, split, image_id = image_id_1, timestamp = timestamp_1, label = label_1),
  `2` = select(pairs, split, image_id = image_id_2, timestamp = timestamp_2, label = label_2),
  .id = "i"
)
splits <- pairs %>%
  select(timestamp, split) %>%
  distinct()

pred <- read_csv(pred_file, show_col_types = FALSE) %>%
  mutate(
    timestamp = with_tz(timestamp, tzone = "US/Eastern"),
    date = as_date(timestamp),
    hour = hour(timestamp),
    rank_obs = rank(flow_cfs, na.last = "keep") / n(),
    rank_pred = rank(score) / n(),
    rank_resid = (rank_obs - rank_pred)
  ) %>%
  left_join(
    splits, by = "timestamp"
  ) %>%
  mutate(
    split = coalesce(split, "na"),
    split = factor(split, levels = c("train", "val", "test", "na"))
  )
pred <- pred %>%
  complete(date = seq.Date(min(pred$date), max(pred$date), by = "1 day")) %>%
  mutate(
    timestamp = coalesce(timestamp, as.POSIXct(date))
  )

pred_day <- pred %>%
  filter(!is.na(score)) %>%
  group_by(date) %>%
  summarise(
    flow_cfs = mean(flow_cfs),
    score = mean(score),
    .groups = "drop"
  ) %>%
  mutate(
    rank_obs = rank(flow_cfs, na.last = "keep") / n(),
    rank_pred = rank(score) / n()
  ) %>%
  left_join(
    pred %>%
      select(date, split) %>%
      filter(!is.na(split), split != "na") %>%
      distinct(),
    by = "date"
  ) %>%
  complete(date = seq.Date(min(pred$date), max(pred$date), by = "1 day")) %>%
  mutate(split = factor(coalesce(split, "na"), levels = levels(pred$split)))


if (any(!is.na(pred$flow_cfs))) {
  safe_cor <- possibly(cor, otherwise = NA_real_)
  tau <- pred %>%
    bind_rows(
      pred %>%
        transmute(
          split = "total",
          flow_cfs,
          score
        )
    ) %>%
    filter(!is.na(score)) %>%
    group_by(split) %>%
    summarise(
      n = sum(!is.na(score)),
      tau = safe_cor(flow_cfs, score, method = "kendall", use = "complete.obs")
    ) %>%
    select(split, tau) %>%
    deframe()

  tau_day <- pred_day %>%
    bind_rows(
      pred_day %>%
        transmute(
          split = "total",
          flow_cfs,
          score
        )
    ) %>%
    filter(!is.na(score)) %>%
    group_by(split) %>%
    summarise(
      n = sum(!is.na(score)),
      tau = safe_cor(flow_cfs, score, method = "kendall", use = "complete.obs")
    ) %>%
    select(split, tau) %>%
    deframe()
} else {
  pred$rank_obs <- NA_real_
  pred$rank_resid <- NA_real_
  pred_day$rank_obs <- NA_real_
  pred_day$rank_resid <- NA_real_

  tau <- list(
    train = NA_real_,
    val = NA_real_,
    test = NA_real_,
    na = NA_real_,
    total = NA_real_
  )
  tau_day <- list(
    train = NA_real_,
    val = NA_real_,
    test = NA_real_,
    na = NA_real_,
    total = NA_real_
  )
}

pred %>%
  filter(!is.na(split)) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = rank_pred, linetype = "Predicted"), color = "black") +
  geom_point(
    aes(y = rank_obs, color = "Observed"), size = 1, alpha = 0.5
  ) +
  scale_color_brewer(NULL, palette = "Set1", labels = split_labels) +
  scale_x_datetime(date_labels = "%b %Y", date_breaks = "2 months", expand = expansion()) +
  scale_y_continuous(limits = c(0, NA), breaks = scales::pretty_breaks(n = 8), labels = scales::percent_format(accuracy = 1), expand = expansion()) +
  guides(
    color = guide_legend(override.aes = list(size = 1, alpha = 1))
  ) +
  labs(
    x = "Date",
    y = "Percentile Rank",
    linetype = NULL
  ) +
  theme_bw() +
  theme(text = element_text(size = 18))



pred %>%
  filter(!is.na(split)) %>%
  ggplot(aes(rank_obs, rank_pred)) +
  geom_point() +
  geom_abline(color = "deepskyblue") +
  scale_color_brewer(NULL, palette = "Set1", labels = split_labels) +
  # scale_x_datetime(date_labels = "%b %Y", date_breaks = "2 months", expand = expansion()) +
  scale_y_continuous(limits = c(0, 1), breaks = scales::pretty_breaks(n = 4), labels = scales::percent_format(accuracy = 1), expand = expansion()) +
  scale_x_continuous(limits = c(0, 1), breaks = scales::pretty_breaks(n = 4), labels = scales::percent_format(accuracy = 1), expand = expansion()) +
  guides(
    color = guide_legend(override.aes = list(size = 1, alpha = 1))
  ) +
  labs(
    x = "Observed Rank",
    y = "Predicted Rank",
    linetype = NULL
  ) +
  theme_bw() +
  theme(aspect.ratio = 1, text = element_text(size = 24), plot.margin = margin(20))




pred %>%
  filter(!is.na(split), year(timestamp) == 2022, month(timestamp) %in% c(9)) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = rank_pred, linetype = "Predicted"), color = "black") +
  geom_point(
    aes(y = rank_obs, color = "Observed"), size = 2, alpha = 0.5
  ) +
  scale_color_brewer(NULL, palette = "Set1", labels = split_labels) +
  scale_x_datetime(date_labels = "%b %d, %Y", date_breaks = "5 days", expand = expansion()) +
  scale_y_continuous(limits = c(0, NA), breaks = scales::pretty_breaks(n = 8), labels = scales::percent_format(accuracy = 1), expand = expansion(mult = c(0.05))) +
  guides(
    color = guide_legend(override.aes = list(size = 1, alpha = 1))
  ) +
  labs(
    x = "Date",
    y = "Percentile Rank",
    linetype = NULL
  ) +
  theme_bw() +
  theme(text = element_text(size = 18), plot.margin = margin(5, 5, 5, 5, unit = "mm"))

pred_day %>%
  filter(!is.na(split)) %>%
  ggplot(aes(date)) +
  geom_line(aes(y = rank_pred, linetype = "Predicted"), color = "black") +
  geom_point(
    aes(y = rank_obs, color = "Observed"), size = 2
  ) +
  scale_color_brewer(NULL, palette = "Set1", labels = split_labels) +
  scale_x_date(date_labels = "%b %Y", date_breaks = "2 months", expand = expansion()) +
  scale_y_continuous(limits = c(0, NA), breaks = scales::pretty_breaks(n = 8), labels = scales::percent_format(accuracy = 1), expand = expansion()) +
  guides(
    color = guide_legend(override.aes = list(size = 1, alpha = 1))
  ) +
  labs(
    x = "Date",
    y = "Percentile Rank",
    linetype = NULL
  ) +
  theme_bw() +
  theme(text = element_text(size = 18), plot.margin = margin(5, 5, 5, 5, unit = "mm"))


pred_day %>%
  filter(!is.na(split)) %>%
  ggplot(aes(rank_obs, rank_pred)) +
  geom_point() +
  geom_abline(color = "deepskyblue", linewidth = 2) +
  scale_color_brewer(NULL, palette = "Set1", labels = split_labels) +
  # scale_x_datetime(date_labels = "%b %Y", date_breaks = "2 months", expand = expansion()) +
  scale_y_continuous(limits = c(0, 1), breaks = scales::pretty_breaks(n = 4), labels = scales::percent_format(accuracy = 1), expand = expansion()) +
  scale_x_continuous(limits = c(0, 1), breaks = scales::pretty_breaks(n = 4), labels = scales::percent_format(accuracy = 1), expand = expansion()) +
  guides(
    color = guide_legend(override.aes = list(size = 1, alpha = 1))
  ) +
  labs(
    x = "Observed Rank",
    y = "Predicted Rank",
    linetype = NULL
  ) +
  theme_bw() +
  theme(aspect.ratio = 1, text = element_text(size = 24), plot.margin = margin(20))
x_day <- pred_day %>%
  filter(!is.na(split)) |>
  select(rank_obs, rank_pred)
cor(x_day$rank_obs, x_day$rank_pred, use = "complete.obs")

x <- pred %>%
  filter(!is.na(split)) |>
  select(rank_obs, rank_pred)
cor(x$rank_obs, x$rank_pred, use = "complete.obs")



# avery inst
library(dataRetrieval)
westb0_nwis <- readNWISuv("01171100", parameterCd = "00060", startDate = min(as_date(pred$timestamp)), endDate = max(as_date(pred$timestamp)))
avery_nwis <- readNWISuv("01171000", parameterCd = "00060", startDate = min(as_date(pred$timestamp)), endDate = max(as_date(pred$timestamp)))

westb0 <- renameNWISColumns(westb0_nwis) |>
  as_tibble() |>
  janitor::clean_names() |>
  select(timestamp = date_time, flow_cfs = flow_inst)
avery <- renameNWISColumns(avery_nwis) |>
  as_tibble() |>
  janitor::clean_names() |>
  select(timestamp = date_time, flow_cfs = flow_inst)


nwis <- bind_rows(
  westb0 = westb0,
  avery = avery,
  .id = "site"
)

pred2 <- pred |>
  mutate(timestamp = round_date(timestamp, unit = "15 minutes")) |>
  left_join(nwis |> pivot_wider(names_from = "site", values_from = "flow_cfs"), by = c("timestamp"))

pred2 |>
  ggplot(aes(flow_cfs, avery)) +
  geom_point()

nwis |>
  ggplot(aes(flow_cfs)) +
  stat_ecdf(aes(color = site)) +
  scale_x_log10()

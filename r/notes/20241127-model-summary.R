# summary of all models

library(tidyverse)
library(glue)

root_dir <- "~/Dropbox/Work/ecosheds/fpe/data/model/20241127/"

db_stations <- read_csv(file.path(root_dir, "fpe-models-stations-20241127.csv"))
db_models <- read_csv(file.path(root_dir, "fpe-models-20241127.csv")) |>
  filter(default)

load_model_predictions <- function(station_id, model_code, root_dir) {
  model_dir <- file.path(root_dir, "rank", station_id, "models", model_code)
  pred_file <- file.path(model_dir, "transform", "predictions.csv")
  pred <- read_csv(pred_file, show_col_types = FALSE)
  return(pred)
}

calculate_pairwise_accuracy <- function(observed, predicted) {
  # Ensure inputs are vectors of the same length
  if (length(observed) != length(predicted)) {
    stop("Observed and predicted must have the same length.")
  }

  # Convert inputs to matrices
  n <- length(observed)

  # Create matrices for all pairwise comparisons
  obs_diff <- outer(observed, observed, FUN = "-")  # Observed differences
  pred_diff <- outer(predicted, predicted, FUN = "-")  # Predicted differences

  # Identify comparable pairs (non-zero observed differences)
  comparable <- obs_diff != 0

  # Check concordant pairs: sign(obs_diff) == sign(pred_diff)
  concordant <- (obs_diff > 0 & pred_diff > 0) | (obs_diff < 0 & pred_diff < 0)

  # Count concordant and comparable pairs
  num_concordant <- sum(concordant & comparable)
  num_comparable <- sum(comparable)

  # Compute pairwise accuracy
  if (num_comparable == 0) {
    return(NA)  # No comparable pairs
  }

  accuracy <- num_concordant / num_comparable
  return(accuracy)
}

ap <- function(relevant, pred_scores) {
  order_pred <- order(pred_scores, decreasing = TRUE)
  relevant <- relevant[order_pred]
  precision <- cumsum(relevant) / seq_along(relevant)
  return(sum(precision * relevant) / sum(relevant))
}

# High flow MAP
calculate_map <- function (values, scores, thresholds = c(0.50, 0.75, 0.90, 0.95), direction = c("HIGH", "LOW")) {
  direction <- match.arg(direction)
  precisions <- sapply(thresholds, function(percentile) {
    threshold <- quantile(values, percentile)
    if (direction == "HIGH") {
      relevant <- values >= threshold
    } else if (direction == "LOW") {
      relevant <- values <= threshold
      scores <- -1 * scores
    } else {
      stop("Invalid direction option")
    }

    ap(relevant, scores)
  })

  mean(precisions)
}

compute_model_stats <- function (value, score) {
  tau <- cor(value, score, method = "kendall")
  rho <- cor(value, score, method = "spearman")
  # pa <- calculate_pairwise_accuracy(value, score)
  rank_value <- (rank(value) - 1) / (length(value) - 1)
  rank_score <- (rank(score) - 1) / (length(score) - 1)
  mae <- mean(abs(rank_value - rank_score))

  map_high_values <- tibble(
    direction = "HIGH",
    threshold = c(seq(0.05, 0.95, by = 0.05), seq(0.96, 0.99, by = 0.01)),
    precision = map_dbl(threshold, \(x) calculate_map(value, score, thresholds = x, direction = "HIGH"))
  )
  # map_high <- mean(map_high_values$precision)
  map_high <- calculate_map(value, score, thresholds = c(0.75, 0.9, 0.95), direction = "HIGH")

  map_low_values <- tibble(
    direction = "LOW",
    threshold = 1 - c(seq(0.05, 0.95, by = 0.05), seq(0.96, 0.99, by = 0.01)),
    precision = map_dbl(threshold, \(x) calculate_map(value, score, thresholds = x, direction = "LOW"))
  )
  # map_low <- mean(map_low_values$precision)
  map_low <- calculate_map(value, score, thresholds = 1 - c(0.75, 0.9, 0.95), direction = "LOW")

  list(
    n = length(value),
    tau = tau,
    rho = rho,
    mae = mae,
    # pa = pa,
    map_high = map_high,
    map_high_values = map_high_values,
    map_low = map_low,
    map_low_values = map_low_values
  )
}

prepare_predictions <- function (x) {
  bind_rows(
    x,
    x |>
      mutate(split = "total")
  ) |>
    filter(!is.na(value), !is.na(score))
}

x <- load_model_predictions(29, "RANK-FLOW-20240424", root_dir) |>
  prepare_predictions() |>
  group_by(split) |>
  summarize(
    stats = list(compute_model_stats(score, value))
  )
x |>
  unnest_wider(stats)


# load all ----------------------------------------------------------------

pred_stats <- db_models |>
  filter(variable_id == "FLOW_CFS") |>
  select(id, station_id, code, uuid, status) |>
  mutate(
    stats = pmap(list(station_id, code), function (station_id, code) {
      print(station_id)
      x <- load_model_predictions(station_id, code, root_dir) |>
        prepare_predictions()
      if (nrow(x) == 0) return(NULL)
      x |>
        nest_by(split) |>
        mutate(
          stats = list(compute_model_stats(data$score, data$value))
        ) |>
        unnest_wider(stats)
    }, .progress = TRUE)
  ) |>
  unnest(stats) |>
  mutate(
    split = factor(split, levels = c("train", "val", "test-in", "test-out", "total"))
  ) |>
  print()

pred_stats |>
  filter(n > 100) |>
  mutate(
    sd_log_value = map_dbl(data, ~ sd(log10(pmax(.$value, 0.01)), na.rm = TRUE)),
    sd_score = map_dbl(data, ~ sd(.$score, na.rm = TRUE)),
  ) |>
  select(-data) |>
  # select(id, station_id, split, map_high_values, map_low_values) |>
  pivot_longer(c(map_high_values, map_low_values)) |>
  unnest(value) |>
  ggplot(aes(threshold, precision, color = sd_log_value)) +
  geom_line(aes(linetype = direction, group = interaction(direction, factor(station_id)))) +
  geom_point() +
  scale_color_viridis_c() +
  facet_wrap(vars(split), ncol = 2)

library(GGally)
pred_stats |>
  filter(n > 100) |>
  select(n, tau, rho, mae, map_high, map_low) |>
  ggpairs()

pred_stats |>
  filter(n > 100) |>
  select(id, station_id, split, tau, rho, mae, map_high, map_low) |>
  pivot_longer(-c(id, station_id, split)) |>
  pivot_wider(names_from = "split") |>
  pivot_longer(-c(id, station_id, name, train), names_to = "split") |>
  ggplot(aes(train, value)) +
  geom_abline() +
  geom_point() +
  facet_wrap(vars(name, split), scales = "free", ncol = 4) +
  theme(aspect.ratio = 1)

library(patchwork)
pred_stats |>
  filter(n > 100) |>
  select(id, station_id, code, uuid, split, data) |>
  unnest(data) |>
  nest_by(split) |>
  mutate(
    p = list({
      data |>
        ggplot(aes(value, score)) +
        geom_hex() +
        scale_x_log10() +
        scale_fill_viridis_c() +
        facet_wrap(vars(station_id), nrow = 1) +
        labs(title = split)
    })
  ) |>
  pull(p) |>
  wrap_plots(ncol = 1)

pred_stats |>
  filter(n > 100, split == "total") |>
  select(id, station_id, code, uuid, split, tau, data) |>
  arrange(desc(tau)) |>
  rowwise() |>
  mutate(
    p = list({
      data |>
        ggplot(aes(value, score)) +
        geom_hex() +
        scale_x_log10() +
        scale_fill_viridis_c() +
        labs(
          title = station_id,
          subtitle = glue("tau = {round(tau, digits = 2)}")
        )
    })
  ) |>
  pull(p) |>
  wrap_plots()

library(ggrepel)
pred_stats |>
  filter(n > 100, split == "total") |>
  mutate(
    cv_value = map_dbl(data, ~ sd(log10(pmax(.$value, 0.01)), na.rm = TRUE) / mean(log10(pmax(.$value, 0.01)), na.rm = TRUE)),
  ) |>
  ggplot(aes(cv_value, map_low)) +
  geom_point(aes(size = n)) +
  geom_text_repel(aes(label = station_id), size = 3)

library(ggrepel)
pred_stats |>
  filter(n > 100, split == "total") |>
  mutate(
    cv_value = map_dbl(data, ~ sd(log10(pmax(.$value, 0.01)), na.rm = TRUE) / mean(log10(pmax(.$value, 0.01)), na.rm = TRUE)),
  ) |>
  arrange(desc(cv_value)) |>
  rowwise() |>
  mutate(
    p = list({
      data |>
        ggplot(aes(timestamp, log10(pmax(value, 0.01)))) +
        geom_line() +
        labs(
          title = station_id,
          subtitle = glue("cv(value) = {round(cv_value, digits = 2)}, tau = {round(tau, digits = 2)}")
        )
    })
  ) |>
  pull(p) |>
  wrap_plots()

pred_stats |>
  filter(n > 100, split == "total") |>
  mutate(
    cv_value = map_dbl(data, ~ sd(.$value, na.rm = TRUE) / mean(.$value, na.rm = TRUE)),
    cv_log_value = map_dbl(data, ~ sd(log10(pmax(.$value, 0.01)), na.rm = TRUE) / mean(log10(pmax(.$value, 0.01)), na.rm = TRUE)),
  ) |>
  ggplot(aes(cv_value, cv_log_value)) +
  geom_point()

pred_stats |>
  filter(n > 100, split == "total") |>
  mutate(
    cv_value = map_dbl(data, ~ sd(.$value, na.rm = TRUE) / mean(.$value, na.rm = TRUE)),
    cv_log_value = map_dbl(data, ~ sd(log10(pmax(.$value, 0.01)), na.rm = TRUE) / mean(log10(pmax(.$value, 0.01)), na.rm = TRUE)),
  ) |>
  select(id, cv_log_value, data) |>
  unnest(data) |>
  ggplot(aes(log10(pmax(value, 0.0001)))) +
  stat_ecdf(aes(group = factor(id), color = cv_log_value)) +
  scale_color_viridis_c()

pred_stats |>
  filter(n > 100, split == "total") |>
  mutate(
    cv_value = map_dbl(data, ~ sd(.$value, na.rm = TRUE) / mean(.$value, na.rm = TRUE)),
    cv_log_value = map_dbl(data, ~ sd(log10(pmax(.$value, 0.01)), na.rm = TRUE) / mean(log10(pmax(.$value, 0.01)), na.rm = TRUE)),
  ) |>
  select(id, cv_log_value, data) |>
  unnest(data) |>
  ggplot(aes(timestamp, log10(pmax(value, 0.0001)))) +
  geom_line(aes(group = factor(id), color = cv_log_value)) +
  scale_color_viridis_c()


pred_stats |>
  filter(n > 100) |>
  ggplot(aes(split, tau)) +
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75)) +
  geom_jitter(aes(color = n), height = 0, width = 0.2) +
  scale_color_viridis_c(trans = "log10")

pred_stats |>
  filter(n > 100) |>
  filter(split == "test-out") |>
  select(id, station_id, code, uuid, split, data) |>
  unnest(data) |>
  group_by(id) |>
  mutate(
    value = log10(pmax(value, 0.01)),
    value = (value - min(value)) / (max(value) - min(value)),
    score = (score - min(score)) / (max(score) - min(score))
  ) |>
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value, color = "obs")) +
  geom_line(aes(y = score, color = "score")) +
  facet_wrap(vars(station_id), ncol = 2, scales = "free")

pred_stats |>
  filter(n > 100) |>
  filter(split == "test-out") |>
  select(id, station_id, code, uuid, split, data) |>
  unnest(data) |>
  group_by(id) |>
  mutate(
    value = (rank(value) - 1) / (n() - 1),
    score = (rank(score) - 1) / (n() - 1)
  ) |>
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value, color = "obs")) +
  geom_line(aes(y = score, color = "score")) +
  facet_wrap(vars(station_id), ncol = 2, scales = "free")

pred_stats |>
  filter(n > 100) |>
  filter(split == "test-out") |>
  select(id, station_id, code, uuid, split, data) |>
  unnest(data) |>
  group_by(id, station_id, timestamp = as_date(timestamp)) |>
  summarise(
    value = mean(value),
    score = mean(score)
  ) |>
  ungroup() |>
  group_by(id) |>
  mutate(
    value = (rank(value) - 1) / (n() - 1),
    score = (rank(score) - 1) / (n() - 1)
  ) |>
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value, color = "obs")) +
  geom_line(aes(y = score, color = "score")) +
  facet_wrap(vars(station_id), ncol = 2, scales = "free")


pred_stats |>
  filter(n > 100) |>
  ggplot(aes(factor(station_id), tau)) +
  geom_point(aes(color = split), size = 4, alpha = 0.5) +
  coord_flip()



# random experiments -------------------------------------------------------


obs_WB0 <- pred_stats |>
  filter(split == "total", station_id == 29) |>
  select(data) |>
  unnest(data) |>
  mutate(log_value = log10(value))


obs_WB0 |>
  ggplot(aes(timestamp, log_value)) +
  geom_line()

obs_WB0 |>
  ggplot(aes(log_value, score)) +
  geom_point()


sim_WB0 <- tibble(
  sd = c(0.01, 0.1, 1)
) |>
  rowwise() |>
  mutate(
    data = list({
      obs_WB0 |>
        mutate(
          score = log10(value) + rnorm(n(), sd = sd)
        )
    }),
    stats = list({
      compute_model_stats(data$value, data$score)
    }),
    p = list(
      data |>
        ggplot(aes(log10(value), score)) +
        geom_point(size = 1)
    )
  )

sim_WB0 |>
  pull(p) |>
  wrap_plots()

sim_WB0 |>
  select(sd, stats) |>
  unnest_wider(stats)


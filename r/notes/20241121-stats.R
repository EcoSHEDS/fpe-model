setwd("~/data/fpe/experiments/20241120-WB0-arch/")

library(tidyverse)
library(jsonlite)
library(glue)
library(yardstick)

output_default <- read_json("runs/default/output/test/output.json")
output_default$summary
pred_default <- read_csv("runs/default/output/test/predictions.csv")



# dev evaluate_rank_predictions ----------------------------------------------------

scores <- pred_default$score
values <- pred_default$value

# Input validation
if (length(scores) != length(values)) {
  stop("scores and values must have the same length")
}

# Rank correlations
kendall_tau <- cor(scores, values, method = "kendall")
stopifnot(output_default$summary$kendall_tau == round(kendall_tau, digits = 4))
spearman_rho <- cor(scores, values, method = "spearman")
stopifnot(output_default$summary$spearman_rho == round(spearman_rho, digits = 4))

kendall_tau
spearman_rho

# Concordance index
c_index <- survcomp::concordance.index(
  x = scores,  # Predicted scores
  surv.time = log10(values),  # Observed flows
  surv.event = rep(1, length(values)),  # Event indicator (use 1 for all cases if no censoring)
  method = "noether"  # Default method (Noether's method)
)

# View the result
c_index$c.index


# Pairwise accuracy
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
pairwise_accuracy <- calculate_pairwise_accuracy(values, scores)


# Mean Absolute Error of Ranks
# rank -> 1 to n, mean of ties
value_ranks <- rank(values) / length(values)
score_ranks <- rank(scores) / length(scores)
rank_mae <- mean(abs(value_ranks - score_ranks))
stopifnot(output_default$summary$rank_mae == round(rank_mae, digits = 4))

rank_mae

# Calculate Average Precision
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

map_high <- calculate_map(values, scores, thresholds = c(0.5, 0.75, 0.9, 0.95), direction = "HIGH")
stopifnot(output_default$summary$map_high == round(map_high, digits = 4))

map_low <- calculate_map(values, scores, thresholds = c(0.5, 0.25, 0.1, 0.05), direction = "LOW")
stopifnot(output_default$summary$map_low == round(map_low, digits = 4))
c(map_low, map_high)

map_high <- calculate_map(values, scores, thresholds = c(0.75, 0.9, 0.95), direction = "HIGH")
map_low <- calculate_map(values, scores, thresholds = c(0.25, 0.1, 0.05), direction = "LOW")
c(map_low, map_high)


map_high_values <- tibble(
  threshold = c(seq(0.05, 0.95, by = 0.05), seq(0.96, 0.99, by = 0.01)),
  precision = map_dbl(threshold, \(x) calculate_map(values, scores, thresholds = x, direction = "HIGH"))
)
map_low_values <- tibble(
  threshold = 1 - c(seq(0.05, 0.95, by = 0.05), seq(0.96, 0.99, by = 0.01)),
  precision = map_dbl(threshold, \(x) calculate_map(values, scores, thresholds = x, direction = "LOW"))
)

bind_rows(
  low = map_low_values,
  high = map_high_values,
  .id = "direction"
) %>%
  ggplot(aes(threshold, precision, color = direction)) +
  geom_line() +
  geom_point()


# pr ----------------------------------------------------------------------

k <- 0.9

# Define high flows as top 10% observed ranks
relevant <- value_ranks >= k  # Boolean: TRUE for relevant items

# Sort data by predicted scores (descending)
ranked_indices <- order(scores, decreasing = TRUE)
sorted_relevant <- relevant[ranked_indices]

# Initialize lists for precision and recall
precision <- c()
recall <- c()
tp <- 0  # True positives
fp <- 0  # False positives

# Total number of relevant items
total_relevant <- sum(relevant)

# Loop through predictions to calculate precision and recall
for (i in seq_along(sorted_relevant)) {
  if (sorted_relevant[i]) {
    tp <- tp + 1  # Increment true positives
  } else {
    fp <- fp + 1  # Increment false positives
  }
  precision[i] <- tp / (tp + fp)  # Precision at this threshold
  recall[i] <- tp / total_relevant  # Recall at this threshold
}

# Combine into a data frame for visualization
pr_curve <- data.frame(Recall = recall, Precision = precision)

pr_curve %>%
  ggplot(aes(recall, precision)) +
  geom_line()

k <- 0.9
yardstick::precision_vec(
  truth = as.factor(value_ranks >= k),
  estimate = as.factor(score_ranks >= k)
)


# pr curve
k <- 0.9
threshold <- quantile(value_ranks, probs = k)  # Top 10%
relevance <- as.factor(value_ranks >= k)  # Binary relevance (TRUE/FALSE)

# Combine into a data frame
data <- data.frame(truth = relevance, estimate = score_ranks)

# Compute PR Curve
pr <- pr_curve(data, truth = truth, estimate) %>%
  filter(is.finite(.threshold))

# View the output
summary(pr)
pr_auc(data, truth = truth, estimate)


pr_curves <- tibble(
  k = seq(0.75, 0.95, by = 0.05),
  data = map(k, \(x) tibble(relevance = as.factor(value_ranks >= x), estimate = score_ranks)),
  curve = map(data, \(x) filter(pr_curve(x, truth = relevance, estimate), is.finite(.threshold))),
  auc = map(data, \(x) pr_auc(x, truth = relevance, estimate))
)

pr_curves %>%
  select(k, auc) %>%
  unnest_wider(auc) %>%
  ggplot(aes(k, .estimate)) +
  geom_line() +
  geom_point()

pr_curves %>%
  select(k, curve) %>%
  unnest(curve) %>%
  ggplot(aes(x = recall, y = precision, color = factor(k))) +
  geom_line() +
  labs(
    title = "Precision-Recall Curve",
    x = "Recall",
    y = "Precision"
  ) +
  theme_minimal()




percentile <- 0.9
data <- tibble(
  truth = values,
  prediction = scores,
  true_rank = rank(truth),
  pred_rank = rank(prediction)
)
calculate_percentile_metrics <- function(data, percentile) {
  # Calculate threshold value from observed flows
  thresh_truth <- quantile(data$truth, percentile)
  thresh_prediction <- quantile(data$prediction, percentile)

  # Add binary indicators for observed and predicted exceedance
  x <- data %>%
    mutate(
      true_exceed = truth >= thresh_truth,
      pred_exceed = prediction >= thresh_prediction
    )

  # Calculate confusion matrix metrics
  metrics <- x %>%
    summarise(
      precision = precision_vec(factor(pred_exceed), factor(true_exceed)),
      recall = recall_vec(factor(pred_exceed), factor(true_exceed)),
      f1 = f_meas_vec(factor(pred_exceed), factor(true_exceed)),
      n_true = sum(true_exceed),
      n_pred = sum(pred_exceed),
      TP = sum(true_exceed & pred_exceed),
      TN = sum(!true_exceed & !pred_exceed),
      FP = sum(!true_exceed & pred_exceed),
      FN = sum(true_exceed & !pred_exceed),
      total = TP + TN + FP + FN
    )

  # Add percentile information
  metrics %>%
    mutate(percentile = percentile, thresh_truth = thresh_truth, thresh_prediction = thresh_prediction)
}


threshold_metrics <- map_dfr(c(0.5, 0.75, 0.9, 0.95), \(x) calculate_percentile_metrics(data, x))

threshold_metrics %>%
  ggplot(aes(percentile, f1)) +
  geom_line()

threshold_metrics <- map_dfr(seq(0.01, 1, by = 0.01), \(x) calculate_percentile_metrics(data, x))

threshold_metrics %>%
  ggplot(aes(percentile)) +
  geom_line(aes(y = precision, color = "precision")) +
  geom_line(aes(y = recall, color = "recall")) +
  geom_line(aes(y = f1, color = "f1"))

threshold_metrics %>%
  ggplot(aes(recall, precision)) +
  geom_point()



percentile <- 0.95
data <- tibble(
  truth = values,
  prediction = scores,
  true_rank = rank(truth) / length(values),
  pred_rank = rank(prediction) / length(scores)
) %>%
  mutate(
    true_class = factor(true_rank >= percentile),
    pred_class = factor(pred_rank >= percentile)
  )
library(janitor)
tabyl(data, true_class, pred_class)
summary(yardstick::conf_mat(data, truth = true_class, pred_class))

yardstick::pr_curve(data, truth = true_class, pred_rank) %>%
  filter(is.finite(.threshold)) %>%
  ggplot(aes(recall, precision)) +
  geom_point(aes(color = .threshold)) +
  scale_color_viridis_c(limits = c(0, 1))


percentile <- 0.9
true_rank <- rank(values) / length(values)
est_rank <- rank(scores) / length(scores)
true_class <- true_rank > percentile
est_class <- est_rank > percentile
TP <- sum(true_class & est_class)
TN <- sum(!true_class & !est_class)
FP <- sum(!true_class & est_class)
FN <- sum(true_class & !est_class)
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

yardstick::precision_vec(factor(true_class), factor(est_class), event_level = "second")
yardstick::recall_vec(factor(true_class), factor(est_class), event_level = "second")
tibble(
  truth = factor(!true_class),
  estimated = factor(!est_class)
) %>%
  yardstick::conf_mat(truth, estimated) %>%
  summary()

calculate_pr <- function (values, estimated, percentile = 0.9, direction = c(">", "<")) {
  direction <- match.arg(direction)
  true_rank <- rank(values) / length(values)
  est_rank <- rank(estimated) / length(estimated)
  if (direction == ">") {
    true_class <- true_rank > percentile
    est_class <- est_rank > percentile
  } else if (direction == "<") {
    true_class <- true_rank < percentile
    est_class <- est_rank < percentile
  } else {
    stop("Invalid direction")
  }
  TP <- sum(true_class & est_class)
  TN <- sum(!true_class & !est_class)
  FP <- sum(!true_class & est_class)
  FN <- sum(true_class & !est_class)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  data.frame(.precision = precision, .recall = recall)
}
calculate_pr(values, scores, percentile = 0.0)

tibble(
  percentile = c(seq(0.05, 0.95, by = 0.05), seq(0.96, 0.99, by = 0.01)),
  pr = map(percentile, \(x) calculate_pr(values, scores, percentile = x))
) %>%
  unnest_wider(pr) %>%
  pivot_longer(-percentile) %>%
  ggplot(aes(percentile, value, color = name)) +
  geom_line() +
  geom_point()


# n_train -----------------------------------------------------------------

load_predictions <- function (x) {
  read_csv(x) %>%
    mutate(
      timestamp = with_tz(timestamp, tzone = "America/New_York"),
      log_value = log10(value),
      score = scale(score)[,1]
    )
}

ntrain <- tibble(
  run_dir = list.dirs("~/data/fpe/experiments/20241111-WB0-n_train/runs", recursive = FALSE),
  run_file = glue("{run_dir}/output/test/predictions.csv")
) %>%
  filter(str_detect(run_dir, "/n_train")) %>%
  mutate(
    n_train = parse_number(basename(run_dir))
  ) %>%
  filter(n_train < 9000) %>%
  arrange(n_train) %>%
  rowwise() %>%
  mutate(
    data = list(load_predictions(run_file))
  ) %>%
  ungroup() %>%
  select(n_train, data)

ntrain_pr <- ntrain %>%
  rowwise() %>%
  mutate(
    pr = list({
      x1 <- tibble(
        percentile = c(seq(0.5, 0.95, by = 0.05), seq(0.96, 0.99, by = 0.01)),
        direction = ">",
        pr = map(percentile, \(x) calculate_pr(data$value, data$score, percentile = x, direction = ">"))
      ) %>%
        unnest_wider(pr)

      x2 <- tibble(
        percentile = 1 - c(seq(0.5, 0.95, by = 0.05), seq(0.96, 0.99, by = 0.01)),
        direction = "<",
        pr = map(percentile, \(x) calculate_pr(data$value, data$score, percentile = x, direction = "<"))
      ) %>%
        unnest_wider(pr)

      bind_rows(x1, x2)
    })
  ) %>%
  unnest(pr) %>%
  print()


ntrain_pr %>%
  select(-data) %>%
  filter(n_train > 200) %>%
#  filter(percentile >= 0.75) %>%
  ggplot(aes(percentile, .precision, color = n_train)) +
  geom_line(aes(group = n_train)) +
  scale_color_viridis_c()

ntrain_pr %>%
  select(-data) %>%
  filter(percentile >= 0.75, n_train > 200) %>%
  ggplot(aes(.recall, .precision)) +
  geom_point()


ntrain_pr %>%
  select(run = n_train, data) %>%
  unnest(data) %>%
  group_by(run, timestamp = as_date(timestamp)) %>%
  summarise(value = mean(value), score = mean(score)) %>%
  ungroup() %>%
  group_by(run) %>%
  mutate(
    value_rank = rank(value) / n(),
    score_rank = rank(score) / n()
  ) %>%
  ungroup() %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value_rank, color = "obs"), linewidth = 2, color = "black") +
  geom_line(aes(y = score_rank, color = run, group = run), alpha = 0.5) +
  scale_color_viridis_c()

ntrain_pr %>%
  select(run = n_train, data) %>%
  unnest(data) %>%
  group_by(run, timestamp = as_date(timestamp)) %>%
  summarise(value = mean(value), score = mean(score)) %>%
  ungroup() %>%
  group_by(run) %>%
  mutate(
    value_rank = rank(value) / n(),
    score_rank = rank(score) / n()
  ) %>%
  ungroup() %>%
  ggplot(aes(score_rank, value_rank)) +
  geom_abline() +
  geom_point(aes(color = run), alpha = 0.5) +
  scale_color_viridis_c()


# transform ---------------------------------------------------------------

transform <- tibble(
  run_dir = list.dirs("~/data/fpe/experiments/20241119-WB0-transform/runs", recursive = FALSE),
  run_file = glue("{run_dir}/output/test/predictions.csv")
) %>%
  filter(file.exists(run_file)) %>%
  rowwise() %>%
  mutate(
    run = basename(run_dir),
    data = list(load_predictions(run_file))
  ) %>%
  ungroup() %>%
  select(run, data)

transform_pr <- transform %>%
  rowwise() %>%
  mutate(
    tau = cor(data$value, data$score, method = "kendall"),
    rho = cor(data$value, data$score, method = "spearman"),
    pr = list({
      x1 <- tibble(
        percentile = c(seq(0.5, 0.95, by = 0.05), seq(0.96, 0.99, by = 0.01)),
        direction = ">",
        pr = map(percentile, \(x) calculate_pr(data$value, data$score, percentile = x, direction = ">"))
      ) %>%
        unnest_wider(pr)

      x2 <- tibble(
        percentile = 1 - c(seq(0.5, 0.95, by = 0.05), seq(0.96, 0.99, by = 0.01)),
        direction = "<",
        pr = map(percentile, \(x) calculate_pr(data$value, data$score, percentile = x, direction = "<"))
      ) %>%
        unnest_wider(pr)

      bind_rows(x1, x2)
    })
  ) %>%
  print()


transform_pr %>%
  arrange(desc(rho))

transform_pr %>%
  ggplot(aes(tau, rho)) +
  geom_abline() +
  geom_point()

transform_pr %>%
  select(-data) %>%
  unnest(pr) %>%
  #filter(percentile >= 0.75) %>%
  ggplot(aes(percentile, .precision, color = run)) +
  geom_line(aes(group = run))

transform_pr %>%
  filter(run %in% c("transform_color_jitter_low", "base", "transform_default", "transform_grayscale")) %>%
  select(run, data) %>%
  unnest(data) %>%
  group_by(run) %>%
  mutate(
    value_rank = rank(value) / n(),
    score_rank = rank(score) / n()
  ) %>%
  ungroup() %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value_rank, color = "obs"), alpha = 0.5, color = "black") +
  geom_line(aes(y = score_rank, color = "est"), alpha = 0.5, color = "red") +
  facet_wrap(vars(run))


transform_pr %>%
  filter(run %in% c("transform_color_jitter_low", "base", "transform_default", "transform_grayscale")) %>%
  select(run, data) %>%
  unnest(data) %>%
  group_by(run, timestamp = as_date(timestamp)) %>%
  summarise(value = mean(value), score = mean(score)) %>%
  ungroup() %>%
  group_by(run) %>%
  mutate(
    value_rank = rank(value) / n(),
    score_rank = rank(score) / n()
  ) %>%
  ungroup() %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value_rank, color = "obs"), alpha = 0.5, color = "black") +
  geom_line(aes(y = score_rank, color = run), alpha = 0.5)


transform_pr %>%
  filter(run %in% c("transform_color_jitter_low", "base", "transform_default", "transform_grayscale")) %>%
  select(run, data) %>%
  unnest(data) %>%
  group_by(run, timestamp = as_date(timestamp)) %>%
  summarise(value = mean(value), score = mean(score)) %>%
  ungroup() %>%
  group_by(run) %>%
  mutate(
    value_rank = rank(value) / n(),
    score_rank = rank(score) / n()
  ) %>%
  ungroup() %>%
  ggplot(aes(score_rank, value_rank)) +
  geom_abline() +
  geom_point(aes(color = run), alpha = 0.5)
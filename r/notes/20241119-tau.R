# explore how tau works

library(tidyvers)
library(slider)

x <- read_csv("~/data/fpe/experiments/20241111-WB0-n_train/runs/n_train_5000/output/test/predictions.csv") %>% 
  mutate(
    timestamp = with_tz(timestamp, tzone = "America/New_York"),
    log_value = log10(value),
    score = scale(score)[,1]
  )

(tau <- cor(x$value, x$score, method = "kendall"))
(spearman_rho <- cor(x$value, x$score, method = "spearman"))
(r_pearson <- cor(x$value, x$score, method = "pearson"))

x %>% 
  ggplot(aes(score, log_value)) +
  geom_point(size = 1, alpha = 0.5)

x %>% 
  ggplot(aes(timestamp, scale(log_value))) +
  geom_line(aes(color = "log10(flow)")) +
  geom_line(aes(y = score, color = "score")) +
  scale_color_brewer(palette = "Set1")

x %>% 
  ggplot(aes(timestamp, scale(log_value))) +
  geom_line(aes(color = "log10(flow)")) +
  geom_line(aes(y = score, color = "score")) +
  scale_color_brewer(palette = "Set1")

x %>% 
  ggplot(aes(timestamp, scale(log_value))) +
  geom_line(aes(color = "log10(flow)")) +
  geom_line(aes(y = score, color = "score"), alpha = 0.25) +
  scale_color_brewer(palette = "Set1")

x %>% 
  ggplot(aes(hour(timestamp) + minute(timestamp) / 60, scale(log_value))) +
  geom_point(aes(color = "log10(flow)")) +
  geom_point(aes(y = score, color = "score"), alpha = 0.25) +
  scale_color_brewer(palette = "Set1") +
  xlim(0, 24)

x %>% 
  ggplot(aes(hour(timestamp) + minute(timestamp) / 60, scale(log_value))) +
  geom_line(aes(group = as_date(timestamp), color = "log10(flow)")) +
  geom_line(aes(group = as_date(timestamp), y = score, color = "score"), alpha = 0.25) +
  scale_color_brewer(palette = "Set1") +
  xlim(0, 24)

x %>% 
  group_by(date = as_date(timestamp)) %>% 
  mutate(
    across(c(score, log_value), ~ . - mean(.))
  ) %>% 
  ungroup() %>% 
  select(timestamp, score, log_value) %>% 
  pivot_longer(-timestamp) %>% 
  ggplot(aes(hour(timestamp) + minute(timestamp) / 60, value)) +
  geom_line(aes(group = as_date(timestamp), color = name), alpha = 0.25) +
  #geom_line(aes(group = as_date(timestamp), color = "log10(flow)")) +
  #geom_line(aes(group = as_date(timestamp), y = score, color = "score"), alpha = 0.25) +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(vars(name), scales = "free_y")

x %>% 
  select(timestamp, score, log_value) %>% 
  mutate(log_value = scale(log_value)) %>% 
  pivot_longer(-timestamp) %>% 
  ggplot(aes(factor(hour(timestamp)), value)) +
  geom_boxplot(aes(fill = name), position = "dodge") +
  scale_fill_brewer(palette = "Set1")

# tau by hour
x %>% 
  group_by(hour = hour(timestamp)) %>% 
  summarize(
    tau = cor(value, score, method = "kendall")
  ) %>% 
  ggplot(aes(hour, tau)) +
  geom_line() +
  geom_point()

x %>% 
  ggplot() +
  aes(log_value, score) +
  geom_point(size = 1, alpha = 0.5) +
  facet_wrap(vars(hour = hour(timestamp)))

# tau by month
x %>% 
  group_by(month = month(timestamp)) %>% 
  summarize(
    tau = cor(value, score, method = "kendall")
  ) %>% 
  ggplot(aes(month, tau)) +
  geom_line() +
  geom_point()

x %>% 
  ggplot() +
  aes(log_value, score) +
  geom_point(size = 1, alpha = 0.5) +
  facet_wrap(vars(month = month(timestamp)))

# by flow bucket
x %>% 
  mutate(q_log_value = rank(log_value) / n()) %>%
  group_by(flow_class = cut(q_log_value, breaks = c(0, 0.25, 0.5, 0.75, 1))) %>% 
  summarize(
    n = n(),
    tau = cor(value, score, method = "kendall")
  ) %>% 
  ggplot(aes(flow_class, tau)) +
  geom_line() +
  geom_point()

x %>% 
  ggplot() +
  aes(log_value, score) +
  geom_point(size = 1, alpha = 0.5) +
  facet_wrap(vars(month = month(timestamp)))


x %>% 
  select(timestamp, log_value, score) %>% 
  pivot_longer(-timestamp) %>% 
  #filter(month(timestamp) == 12) %>% 
  ggplot() +
  aes(timestamp, value, color = name) +
  geom_point(size = 1, alpha = 0.5) +
  facet_wrap(vars(month = month(timestamp)))


# compare n_train ---------------------------------------------------------

load_predictions <- function (x) {
  read_csv(x) %>% 
    mutate(
      timestamp = with_tz(timestamp, tzone = "America/New_York"),
      log_value = log10(value),
      score = scale(score)[,1]
    )
}

x_500 <- load_predictions("~/data/fpe/experiments/20241111-WB0-n_train/runs/n_train_500/output/test/predictions.csv")
x_1000 <- load_predictions("~/data/fpe/experiments/20241111-WB0-n_train/runs/n_train_1000/output/test/predictions.csv")
x_5000 <- load_predictions("~/data/fpe/experiments/20241111-WB0-n_train/runs/n_train_5000/output/test/predictions.csv")
x_10000 <- load_predictions("~/data/fpe/experiments/20241111-WB0-n_train/runs/n_train_10000/output/test/predictions.csv")

x_all <- bind_rows(
  `500` = x_500,
  `1000` = x_1000,
  `5000` = x_5000,
  `10000` = x_10000,
  .id = "n_train"
) %>% 
  mutate(n_train = fct_inorder(n_train))

x_all %>% 
  ggplot(aes(score, log_value)) +
  geom_point(size = 1, alpha = 0.5) +
  facet_wrap(vars(n_train)) +
  theme(aspect.ratio = 1)

x_all %>% 
  ggplot(aes(timestamp, score)) +
  geom_line(aes(color = n_train)) +
  geom_line(
    data = x_500,
    aes(y = scale(log_value))
  ) +
  facet_wrap(vars(n_train))

x_all %>% 
  group_by(n_train) %>% 
  mutate(
    score_24 = slide_dbl(score, mean, .before = 11, .after = 12)
  ) %>% 
  filter(month(timestamp) %in% 6:8) %>% 
  group_by(n_train) %>% 
  mutate(
    score_24 = scale(score_24)
  ) %>% 
  ggplot(aes(timestamp, score_24)) +
  geom_line(aes(color = n_train)) +
  geom_line(
    data = x_500 %>%
      filter(month(timestamp) %in% 6:8),
    aes(y = scale(log_value))
  ) +
  facet_wrap(vars(n_train))

x_all %>% 
  group_by(n_train) %>% 
  mutate(
    score_24 = slide_dbl(score, mean, .before = 23)
  ) %>% 
  ggplot(aes(score_24, log_value)) +
  geom_point(size = 1, alpha = 0.5) +
  facet_wrap(vars(n_train)) +
  theme(aspect.ratio = 1)


x_all %>% 
  group_by(n_train) %>% 
  mutate(q_log_value = rank(log_value) / n()) %>%
  group_by(n_train, flow_class = cut(q_log_value, breaks = c(0, 0.25, 0.5, 0.75, 1))) %>% 
  summarize(
    n = n(),
    tau = cor(value, score, method = "kendall")
  ) %>% 
  ggplot(aes(flow_class, tau, fill = n_train)) +
  geom_col(position = "dodge")


# all n_train -------------------------------------------------------------

all_runs <- tibble(
  run_dir = list.dirs("~/data/fpe/experiments/20241111-WB0-n_train/runs", recursive = FALSE),
  run_file = glue("{run_dir}/output/test/predictions.csv")
) %>% 
  filter(str_detect(run_dir, "/n_train")) %>% 
  mutate(
    n_train = parse_number(basename(run_dir))
  ) %>% 
  filter(n_train < 9000) %>% 
  arrange(n_train) %>% 
  #mutate(n_train = fct_inorder(as.character(n_train))) %>% 
  rowwise() %>% 
  mutate(
    data = list(load_predictions(run_file))
  ) %>% 
  ungroup() %>% 
  select(n_train, data)

all_runs %>% 
  rowwise() %>% 
  mutate(
    tau = cor(data$value, data$score, method = "kendall")
  ) %>% 
  ggplot(aes(n_train, tau)) +
  geom_point() +
  scale_color_viridis_c()

all_runs %>% 
  unnest(data) %>% 
  group_by(n_train, month = month(timestamp)) %>% 
  summarize(
    tau = cor(value, score, method = "kendall")
  ) %>% 
  ggplot(aes(month, tau, color = n_train)) +
  geom_line(aes(group = n_train)) +
  geom_point() +
  scale_color_viridis_c()

all_runs %>% 
  unnest(data) %>% 
  group_by(n_train, month = month(timestamp)) %>% 
  summarize(
    tau = cor(value, score, method = "kendall")
  ) %>% 
  ggplot(aes(n_train, tau, color = factor(month))) +
  geom_line(aes(group = month)) +
  geom_point()

all_runs %>% 
  unnest(data) %>% 
  group_by(n_train, hour = hour(timestamp)) %>% 
  summarize(
    tau = cor(value, score, method = "kendall")
  ) %>% 
  ggplot(aes(hour, tau, color = n_train)) +
  geom_line(aes(group = n_train)) +
  geom_point() +
  scale_color_viridis_c()

all_runs %>% 
  unnest(data) %>% 
  group_by(n_train, hour = hour(timestamp)) %>% 
  summarize(
    tau = cor(value, score, method = "kendall")
  ) %>% 
  ggplot(aes(n_train, tau, color = factor(hour))) +
  geom_line(aes(group = hour)) +
  geom_point() +
  scale_color_viridis_d()

all_runs %>% 
  unnest(data) %>% 
  ggplot() +
  aes(log_value, score) +
  geom_point(size = 1, alpha = 0.5) +
  facet_wrap(vars(n_train))


# by flow bucket
all_runs %>% 
  unnest(data) %>% 
  group_by(n_train) %>% 
  mutate(q_log_value = rank(log_value) / n()) %>%
  group_by(n_train, flow_class = cut(q_log_value, breaks = c(0, 0.25, 0.5, 0.75, 1))) %>% 
  summarize(
    n = n(),
    tau = cor(value, score, method = "kendall")
  ) %>% 
  ggplot(aes(flow_class, tau, color = n_train)) +
  geom_line() +
  geom_point() +
  scale_color_viridis_c()


all_runs %>% 
  unnest(data) %>% 
  group_by(n_train) %>% 
  mutate(
    q_log_value = rank(log_value) / n(),
    #flow_class = cut(q_log_value, breaks = c(0, 0.25, 0.5, 0.75, 1))
  ) %>%
  filter(q_log_value > 0.75) %>% 
  ggplot() +
  aes(log_value, score) +
  geom_point(size = 1, alpha = 0.5) +
  facet_wrap(vars(n_train), scales = "free")


# compare model fit to image quality

library(tidyverse)

images <- read_csv("~/data/fpe/sites/WESTB0/models/RANK-FLOW-20241111/images_quality.csv")
predictions <- read_csv("~/data/fpe/experiments/20241120-WB0-transform/runs/transform_default/output/test/predictions.csv")


df <- predictions %>%
  select(image_id, timestamp, value, score) %>%
  left_join(
    images %>%
      select(image_id, is_grayscale:entropy),
    by = "image_id"
  ) %>%
  mutate(
    timestamp = with_tz(timestamp, tzone = "US/Eastern"),
    dhour = hour(timestamp) + minute(timestamp) / 60,
    rank_value = (rank(value) - 1) / (n() - 1),
    rank_score = (rank(score) - 1) / (n() - 1),
    rank_resid = rank_score - rank_value
  )


summary(df)

df %>%
  ggplot(aes(timestamp, rank_resid)) +
  geom_line()

df %>%
  ggplot(aes(dhour, rank_resid)) +
  geom_point(size = 1, alpha = 0.5) +
  geom_smooth(se = FALSE) +
  facet_wrap(vars(month(timestamp)))

df %>%
  ggplot(aes(factor(is_grayscale), rank_resid)) +
  geom_violin()

df %>%
  rename(obs = value) %>%
  select(-michelson_contrast) %>%
  pivot_longer(c(mean_r:entropy)) %>%
  ggplot(aes(rank_resid, value)) +
  geom_hex() +
  scale_fill_viridis_c() +
  facet_wrap(vars(name), scales = "free", nrow = 4) +
  theme(aspect.ratio = 1)

df %>%
  ggplot(aes(timestamp, rank_resid)) +
  geom_point(aes(color = entropy)) +
  scale_color_viridis_c()


df %>%
  filter(epiweek(timestamp) %in% 30:33) %>%
  mutate(
    # rank_value = (rank(value) - 1) / (n() - 1),
    # rank_score = (rank(score) - 1) / (n() - 1),
    # rank_resid = rank_score - rank_value
  ) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = rank_value, linetype = "obs")) +
  geom_line(aes(y = rank_score, linetype = "score")) +
  geom_point(aes(y = rank_score, color = entropy)) +
  scale_color_viridis_c()

df %>%
  filter(epiweek(timestamp) %in% 30:33) %>%
  mutate(
    # rank_value = (rank(value) - 1) / (n() - 1),
    # rank_score = (rank(score) - 1) / (n() - 1),
    # rank_resid = rank_score - rank_value
  ) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = rank_value, linetype = "obs")) +
  geom_line(aes(y = rank_score, linetype = "score")) +
  geom_point(aes(y = rank_score, color = frequency_energy)) +
  scale_color_viridis_c() +
  facet_wrap(vars(as_date(timestamp)), scales = "free")

df %>%
  filter(epiweek(timestamp) %in% 30:33) %>%
  mutate(
    # rank_value = (rank(value) - 1) / (n() - 1),
    # rank_score = (rank(score) - 1) / (n() - 1),
    # rank_resid = rank_score - rank_value
  ) %>%
  ggplot(aes(dhour)) +
  # geom_line(aes(y = rank_value, linetype = "obs")) +
  # geom_line(aes(y = rank_score, linetype = "score")) +
  geom_point(aes(y = entropy, color = entropy)) +
  scale_color_viridis_c()


x <- df %>%
  filter(epiweek(timestamp) %in% 30:33) %>%
  mutate(
    rank_value = (rank(value) - 1) / (n() - 1),
    rank_score = (rank(score) - 1) / (n() - 1),
    rank_resid = rank_score - rank_value
  )
cor(x$rank_value, x$rank_score, method = "spearman")
x %>%
  rename(obs = value) %>%
  select(-michelson_contrast) %>%
  pivot_longer(c(mean_r:entropy)) %>%
  group_by(name) %>%
  mutate(value = scale(value)) %>%
  ggplot() +
  aes(rank_value, rank_score) +
  geom_abline() +
  geom_point(aes(color = value), size = 1) +
  scale_color_viridis_c() +
  facet_wrap(vars(name))

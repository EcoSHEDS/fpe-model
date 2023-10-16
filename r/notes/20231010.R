# relative -> absolute flow problem

library(tidyverse)
library(jpeg)
library(grid)
library(httr)
library(glue)
library(patchwork)

view_image <- function(x) {
  response <- GET(URLencode(x))

  if (http_type(response) == "binary/octet-stream") {
    # Read the image
    img <- readJPEG(content(response, "raw"))

    # Display the image
    grid.newpage()
    grid.raster(img)
  } else {
    print(glue("Failed to fetch the image or the content type is not JPEG. ({http_type(response)})"))
  }
}

pred <- read_csv("/Users/jeff/data/fpe-model/WESTB0/models/predictions.csv") |>
  mutate(
    date = as_date(timestamp),
    rank_flow = rank(flow_cfs) / n(),
    rank_score = rank(score) / n(),
    pred_flow = approx(rank_flow, flow_cfs, xout = rank_score, method = "linear")$y
  ) |>
  complete(
    date = seq.Date(ymd(20220201), ymd(20230906), by = 1)
  ) |>
  mutate(
    timestamp = coalesce(timestamp, date)
  )

# lowest flow
pred |>
  arrange(flow_cfs) |>
  head(1) |>
  pull(url) |>
  view_image()

# highest flow
pred |>
  arrange(desc(flow_cfs)) |>
  head(1) |>
  pull(url) |>
  view_image()

# median flow
median(pred$score, na.rm = TRUE)
pred |>
  filter(abs(score - median(score, na.rm = TRUE)) < 0.1, year(timestamp) == 2023, month(timestamp) == 6) |>
  head(1) |>
  pull(url) |>
  view_image()


pred |>
  transmute(timestamp, `log10(flow [cfs])` = log10(pred_flow), score) |>
  pivot_longer(-timestamp) |>
  mutate(name = fct_rev(name)) |>
  ggplot(aes(timestamp, value)) +
  geom_line() +
  geom_line(
    data = pred |>
      transmute(timestamp, `log10(flow [cfs])` = log10(flow_cfs)) |>
      pivot_longer(-timestamp) |>
      mutate(name = fct_rev(name)),
    color = "orangered"
  ) +
  facet_wrap(vars(name), scales = "free", ncol = 1, strip.position = "left") +
  labs(y = NULL) +
  theme_bw() +
  theme(
    strip.background = element_blank(),
    strip.placement = "outside",
    strip.text = element_text(size = 12)
  )



p1 <- pred |>
  ggplot(aes(rank_score, score)) +
  geom_line() +
  labs(x = NULL) +
  theme_bw()

p2 <- pred |>
  ggplot(aes(rank_flow, log10(flow_cfs))) +
  geom_line() +
  labs(x = "percentile rank") +
  theme_bw()

p1 / p2



pred_2023 <- pred |>
  filter(year(timestamp) == 2023) |>
  mutate(
    rank_flow = rank(flow_cfs, na.last = "keep") / n(),
    rank_score = rank(score, na.last = "keep") / n(),
    pred_flow_wrong = approx(pred$rank_flow, pred$flow_cfs, xout = rank_score, method = "linear")$y,
    pred_flow = approx(rank_flow, flow_cfs, xout = rank_score, method = "linear")$y
  ) |>
  complete(
    date = seq.Date(ymd(20230101), ymd(20230906), by = 1)
  ) |>
  mutate(
    timestamp = coalesce(timestamp, date)
  )


pred |>
  transmute(timestamp, `log10(flow [cfs])` = log10(pred_flow), score) |>
  pivot_longer(-timestamp) |>
  mutate(name = fct_rev(name)) |>
  ggplot(aes(timestamp, value)) +
  geom_line() +
  geom_line(
    data = pred_2023 |>
      transmute(timestamp, `log10(flow [cfs])` = log10(pred_flow), score) |>
      pivot_longer(-timestamp) |>
      mutate(name = fct_rev(name)),
    color = "deepskyblue"
  ) +
  geom_line(
    data = pred |>
      transmute(timestamp, `log10(flow [cfs])` = log10(flow_cfs)) |>
      pivot_longer(-timestamp) |>
      mutate(name = fct_rev(name)),
    color = "orangered"
  ) +
  annotate(
    geom = "rect",
    xmin = min(pred$timestamp), xmax = ymd_hm(202301010000), ymin = -Inf, ymax = Inf,
    fill = "black", alpha = 0.9
  ) +
  facet_wrap(vars(name), scales = "free", ncol = 1, strip.position = "left") +
  labs(y = NULL) +
  theme_bw() +
  theme(
    strip.background = element_blank(),
    strip.placement = "outside",
    strip.text = element_text(size = 12)
  )


p1 <- pred |>
  ggplot(aes(rank_score, score)) +
  geom_line() +
  geom_line(data = pred_2023, color = "deepskyblue") +
  labs(x = NULL) +
  theme_bw()

p2 <- pred |>
  ggplot(aes(rank_flow, log10(flow_cfs))) +
  geom_line(color = "black", linewidth = 1.5) +
  geom_line(color = "deepskyblue") +
  geom_line(data = pred_2023, color = "deepskyblue", alpha = 0) +
  labs(x = "percentile rank") +
  theme_bw()

p1 / p2


pred |>
  ggplot(aes(rank_flow, log10(flow_cfs))) +
  geom_line(color = "black") +
  geom_line(aes(y = log10(flow_cfs) + 0.4), color = "orangered") +
  labs(x = "percentile rank") +
  theme_bw()

pred |>
  ggplot(aes(rank_flow, log10(flow_cfs))) +
  geom_line(color = "black") +
  geom_line(aes(y = log10(flow_cfs) + 0.4), color = "orangered") +
  geom_line(aes(y = log10(flow_cfs) * 1.5 - 0.15), color = "deepskyblue") +
  labs(x = "percentile rank") +
  theme_bw()



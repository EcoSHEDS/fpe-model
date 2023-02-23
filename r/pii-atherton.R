library(tidyverse)
library(jsonlite)
library(janitor)
library(caret)
library(imager)

x <- read_json("../results/atherton-20230215-v5a-output.json")
x$detection_categories
y <- map_df(x$images, as_tibble) |>
  unnest_wider(detections) |>
  group_by(file) |>
  slice(1)

y |>
  ggplot(aes(conf)) +
  geom_histogram() +
  facet_wrap(vars(category))

y_person <- y |>
  filter(category == 2)

obs <- read_csv("~/data/fpe/pii/atherton-20230215/Atherton_20230215.csv", show_col_types = FALSE)

pred <- obs |>
  select(filename = filename_new, obs = People) |>
  left_join(
    y_person |>
    select(filename = file, prob = conf),
    by = "filename"
  ) |>
  mutate(
    pred = coalesce(prob > 0.005, FALSE),
    obs = factor(obs, levels = c("TRUE", "FALSE")),
    pred = factor(pred, levels = c("TRUE", "FALSE"))
  )
confusionMatrix(pred$obs, pred$pred, mode = "prec_recall")

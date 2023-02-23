library(tidyverse)
library(jsonlite)
library(janitor)
library(caret)
library(imager)

atherton_pred <- read_json("~/data/fpe/pii/atherton-20230215/atherton-20230215-mdv5a-output.json")
atherton_pred$detection_categories

atherton_pred_top <- map_df(atherton_pred$images, as_tibble) |>
  unnest_wider(detections) |>
  group_by(file) |>
  slice(1)

atherton_pred_top |>
  ggplot(aes(conf)) +
  geom_histogram() +
  facet_wrap(vars(category))

atherton_pred_top_person <- atherton_pred_top |>
  filter(category == 2)

atherton_obs <- read_csv("~/data/fpe/pii/atherton-20230215/Atherton_20230215.csv", show_col_types = FALSE)

atherton <- atherton_obs |>
  select(filename = filename_new, obs = People) |>
  left_join(
    atherton_pred_top_person |>
    select(filename = file, prob = conf),
    by = "filename"
  ) |>
  mutate(
    obs = factor(obs, levels = c("TRUE", "FALSE")),
    pred = coalesce(prob > 0.005, FALSE),
    pred = factor(pred, levels = c("TRUE", "FALSE"))
  )

cm <- confusionMatrix(atherton$obs, atherton$pred, mode = "prec_recall")
cm

atherton_cutoff <- tibble(
  cutoff = c(seq(0.0001, 0.001, by = 0.0001), seq(0.001, 0.01, by = 0.001), seq(0.01, 0.1, by = 0.01), seq(0.1, 0.5, by = 0.1))
) |>
  mutate(
    cm = map(cutoff, function (x) {
      y <- atherton |> 
        mutate(
          pred = coalesce(prob > x, FALSE),
          pred = factor(pred, levels = c("TRUE", "FALSE"))
        )
      confusionMatrix(y$obs, y$pred, mode = "prec_recall")
    }),
    cm_stats = map(cm, ~ .$byClass)
  ) |> 
  unnest_wider(cm_stats) |> 
  clean_names()

atherton_cutoff |>
  select(cutoff, precision, recall, f1) |> 
  pivot_longer(-cutoff) |> 
  ggplot(aes(cutoff, value)) + 
  geom_line(aes(color = name)) +
  scale_x_log10(labels = scales::comma) +
  ylim(0, 1)

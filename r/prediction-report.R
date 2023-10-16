generate_report <- function (site, model_id, root_dir = "D:/fpe/sites") {
  output_file <- paste0("./qmd/rank-", site, "-", model_id, ".html")

  quarto::quarto_render(
    input = "qmd/rank-predictions.qmd",
    output_format = "html",
    # output_file = output_file,
    execute_params = list(
      root_dir = root_dir,
      site = site,
      model_id = model_id
    ),
    debug = TRUE
  )
  target_filename <- file.path(
    root_dir, site, "models", model_id,
    paste0("rank-", site, "-", model_id, ".html")
  )
  file.copy("qmd/rank-predictions.html", target_filename, overwrite = TRUE)
  print(paste0("Report saved: ", target_filename))
}

generate_report("WESTB0", "20230922")

# WESTB0: 20230920, 20230921, 20230922 annotation quality --------------------
# 20230920: all annotations
# 20230921: exclude user_id=9b
# 20230922: add 2500 high quality annotations

x <- bind_rows(
  `20230920` = read_csv("D:/fpe/sites/WESTB0/models/20230920/transform/predictions.csv"),
  `20230921` = read_csv("D:/fpe/sites/WESTB0/models/20230921/transform/predictions.csv"),
  `20230922` = read_csv("D:/fpe/sites/WESTB0/models/20230922/transform/predictions.csv"),
  .id = "run"
) %>%
  group_by(run) %>%
  mutate(
    rank_obs = rank(flow_cfs) / n(),
    rank_pred = rank(score) / n()
  )

x %>%
  filter(year(timestamp) == 2022, month(timestamp) == 9) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = rank_pred, linetype = "Predicted")) +
  geom_point(aes(y = rank_obs, color = "Observed"), alpha = 0.5) +
  facet_wrap(vars(run), ncol = 1) +
  labs(y = "rank", linetype = NULL, color = NULL, subtitle = "WESTB0 | Sep 2022") +
  theme_bw()


x %>%
  ggplot(aes(rank_obs, rank_pred)) +
  geom_point(aes(), alpha = 0.5, size = 0.5, color = "orangered") +
  geom_abline() +
  facet_wrap(vars(run), nrow = 1) +
  labs(y = "observed rank", y = "predicted rank", linetype = NULL, color = NULL, subtitle = "WESTB0") +
  theme_bw() +
  theme(aspect.ratio = 1)




# WESTB0: 20230922 annotation subsets ----------------------------------------
# 20230922: all annotations
# 20230922-25: 25% of all annotations
# 20230922-50: 50% of all annotations
# 20230922-75: 75% of all annotations

x_metrics <- bind_rows(
  `25%` = read_csv("D:/fpe/sites/WESTB0/models/20230922-25/output/metrics.csv"),
  `50%` = read_csv("D:/fpe/sites/WESTB0/models/20230922-50/output/metrics.csv"),
  `75%` = read_csv("D:/fpe/sites/WESTB0/models/20230922-75/output/metrics.csv"),
  `100%` = read_csv("D:/fpe/sites/WESTB0/models/20230922/output/metrics.csv"),
  .id = "run"
) %>%
  mutate(run = fct_inorder(run)) %>%
  pivot_longer(-c(run, epoch)) %>%
  mutate(name = str_replace(name, "_loss", "")) %>%
  mutate(name = fct_inorder(name))

x_metrics %>%
  ggplot(aes(epoch, value)) +
  geom_line(aes(color = name)) +
  geom_point(aes(color = name)) +
  facet_wrap(vars(run)) +
  scale_color_brewer(palette = "Set1") +
  ylim(0, NA) +
  labs(y = "loss", color = "split") +
  theme_bw()

x_metrics %>%
  filter(name == "test") %>%
  group_by(run) %>%
  filter(value == min(value)) %>%
  ggplot(aes(run, value)) +
  geom_col() +
  labs(y = "test loss", x = "subset") +
  theme_bw()


x_pairs_100 <- bind_rows(
  train = read_csv("D:/fpe/sites/WESTB0/models/20230922/input/pairs-train.csv"),
  val = read_csv("D:/fpe/sites/WESTB0/models/20230922/input/pairs-val.csv"),
  test = read_csv("D:/fpe/sites/WESTB0/models/20230922/input/pairs-test.csv"),
  .id = "split"
) %>%
  mutate(split = fct_inorder(split), run = "100%")
x_pairs_25 <- bind_rows(
  train = read_csv("D:/fpe/sites/WESTB0/models/20230922-25/input/pairs-train.csv"),
  val = read_csv("D:/fpe/sites/WESTB0/models/20230922-25/input/pairs-val.csv"),
  test = read_csv("D:/fpe/sites/WESTB0/models/20230922-25/input/pairs-test.csv"),
  .id = "split"
) %>%
  mutate(split = fct_inorder(split), run = "25%")
x_pairs_50 <- bind_rows(
  train = read_csv("D:/fpe/sites/WESTB0/models/20230922-50/input/pairs-train.csv"),
  val = read_csv("D:/fpe/sites/WESTB0/models/20230922-50/input/pairs-val.csv"),
  test = read_csv("D:/fpe/sites/WESTB0/models/20230922-50/input/pairs-test.csv"),
  .id = "split"
) %>%
  mutate(split = fct_inorder(split), run = "50%")
x_pairs_75 <- bind_rows(
  train = read_csv("D:/fpe/sites/WESTB0/models/20230922-75/input/pairs-train.csv"),
  val = read_csv("D:/fpe/sites/WESTB0/models/20230922-75/input/pairs-val.csv"),
  test = read_csv("D:/fpe/sites/WESTB0/models/20230922-75/input/pairs-test.csv"),
  .id = "split"
) %>%
  mutate(split = fct_inorder(split), run = "75%")

x_pairs <- bind_rows(x_pairs_25, x_pairs_50, x_pairs_75, x_pairs_100) %>%
  mutate(run = fct_inorder(run))

x_image_splits <- x_pairs %>%
  select(run, split, starts_with("image_id")) %>%
  pivot_longer(-c(run, split), values_to = "image_id") %>%
  select(-name) %>%
  group_by(run, image_id) %>%
  slice(1) %>%
  ungroup()

x <- bind_rows(
  `25%` = read_csv("D:/fpe/sites/WESTB0/models/20230922-25/transform/predictions.csv"),
  `50%` = read_csv("D:/fpe/sites/WESTB0/models/20230922-50/transform/predictions.csv"),
  `75%` = read_csv("D:/fpe/sites/WESTB0/models/20230922-75/transform/predictions.csv"),
  `100%` = read_csv("D:/fpe/sites/WESTB0/models/20230922/transform/predictions.csv"),
  .id = "run"
) %>%
  mutate(run = fct_inorder(run)) %>%
  group_by(run) %>%
  mutate(
    rank_obs = rank(flow_cfs) / n(),
    rank_pred = rank(score) / n()
  ) %>%
  ungroup() %>%
  left_join(x_image_splits, by = c("run", "image_id")) %>%
  mutate(split = factor(coalesce(split, "N/A"), levels = c("train", "val", "test", "N/A")))

janitor::tabyl(x, run, split)

x %>%
  filter(year(timestamp) == 2022, month(timestamp) %in% 4:6) %>%
  arrange(desc(split), timestamp) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = rank_pred, linetype = "Predicted")) +
  geom_point(aes(y = rank_obs, color = split), alpha = 0.5) +
  scale_color_manual(values = c(RColorBrewer::brewer.pal(3, "Set1"), "#AAAAAA"), drop = FALSE) +
  ylim(0, NA) +
  facet_wrap(vars(run), ncol = 1) +
  labs(y = "rank", linetype = NULL, color = NULL, subtitle = "WESTB0 | Apr-Jun 2022") +
  theme_bw()

x %>%
  filter(year(timestamp) == 2023) %>%
  ggplot(aes(timestamp)) +
  geom_line(aes(y = rank_pred, linetype = "Predicted")) +
  geom_point(aes(y = rank_obs, color = split), alpha = 0.5) +
  scale_color_manual(values = c(RColorBrewer::brewer.pal(3, "Set1"), "#AAAAAA"), drop = FALSE) +
  ylim(0, NA) +
  facet_wrap(vars(run), ncol = 1) +
  labs(y = "rank", linetype = NULL, color = NULL, subtitle = "WESTB0 | 2023") +
  theme_bw()

x %>%
  ggplot(aes(rank_obs, rank_pred)) +
  geom_point(aes(color = split), alpha = 0.5, size = 0.5) +
  geom_abline() +
  scale_color_manual(values = c(RColorBrewer::brewer.pal(3, "Set1"), "#AAAAAA"), drop = FALSE) +
  facet_grid(vars(split), vars(run)) +
  labs(x = "observed rank", y = "predicted rank", linetype = NULL, color = "split", subtitle = "WESTB0") +
  theme_bw() +
  theme(aspect.ratio = 1)


library(tidyverse)
library(glue)
library(ggrepel)
library(janitor)
library(patchwork)

x <- map_df(station_ids, \(x) {
  read_csv(glue("/mnt/d/fpe/rank/{x}/models/RANK-FLOW-20240410/transform/predictions.csv")) %>%
    mutate(station_id = x, .before = everything()) %>%
    select(station_id, split, image_id, timestamp, value, score)
}) %>%
  nest_by(station_id)

x2 <- x %>%
  mutate(
    stats = list({
      y <- data %>%
        filter(!is.na(value), !is.na(score))
      y <- y %>%
        bind_rows(
          y %>%
            filter(split != "test-out") %>%
            mutate(split = "all-in"),
          y %>%
            mutate(split = "all")
        )

      y %>%
        group_by(split) %>%
        summarise(
          n = n(),
          value_s = sd(log10(pmax(value, 0.01)), na.rm = TRUE),
          value_m = mean(log10(pmax(value, 0.01)), na.rm = TRUE),
          value_cv = value_s / value_m,
          score_s = sd(score, na.rm = TRUE),
          score_m = mean(score, na.rm = TRUE),
          score_cv = score_s / score_m,
          r2 = cor(log10(pmax(value, 0.01)), score, method = "pearson") ^ 2,
          tau = cor(value, score, method = "kendall")
        )
    })
    # p = list({
    #   data %>%
    #     filter(!is.na(value), !is.na(score)) %>%
    #     slice_sample(frac = 0.1) %>%
    #     mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out"))) %>%
    #     ggplot(aes(value, score)) +
    #     geom_point(aes(color = split), size = 0.5, alpha = 0.5) +
    #     scale_color_brewer(palette = "Set1") +
    #     scale_x_log10() +
    #     labs(title = station_id, subtitle = paste0("tau = ", sprintf("%.2f", tau)))
    # })
  ) %>%
  select(-data)

# higher cv(value) -> higher tau
# less flow variability is harder to model
x2 %>%
  unnest(stats) %>%
  ggplot(aes(value_cv, tau)) +
  geom_point() +
  facet_wrap(vars(split))

# higher sd(score) -> higher tau
# model learns greater variations in score to distinguish photos
# model is more "discerning"
x2 %>%
  unnest(stats) %>%
  ggplot(aes(score_s, tau)) +
  geom_point() +
  facet_wrap(vars(split))

# no effect of sample size
x2 %>%
  unnest(stats) %>%
  ggplot(aes(n, tau)) +
  geom_point() +
  facet_wrap(vars(split), scales = "free_x")

x2 %>%
  unnest(stats) %>%
  select(station_id, split, tau) %>%
  pivot_wider(names_from = "split", values_from = "tau") %>%
  ungroup() %>%
  select(-station_id) %>%
  GGally::ggpairs()

x2 %>%
  pull(p) %>%
  wrap_plots()

# generate uuids
x2 %>%
  select(station_id) %>%
  mutate(uuid = map_chr(station_id, ~ uuid::UUIDgenerate())) %>%
  write_delim("/mnt/d/fpe/rank/stations-wb-models-uuid.txt", col_names = FALSE, delim = " ")

# run fpe-model/batch-deploy.sh to upload report and predictions to S3

# generate file to import to db models table
x3 <- read_delim("/mnt/d/fpe/rank/stations-wb-models-uuid.txt", col_names = c("station_id", "uuid"), delim = " ") %>%
  transmute(
    station_id,
    model_type_id = "RANK",
    variable_id = "FLOW_CFS",
    code = "RANK-FLOW-20240410",
    uuid,
    default = TRUE,
    diagnostics_url = glue("https://usgs-chs-conte-prod-fpe-storage.s3.us-west-2.amazonaws.com/models/{uuid}/{code}.html"),
    predictions_url = glue("https://usgs-chs-conte-prod-fpe-storage.s3.us-west-2.amazonaws.com/models/{uuid}/predictions.csv"),
    status = "DONE"
  ) %>%
  print()
x3 %>%
  write_csv("/mnt/d/fpe/rank/stations-wb-models-db.csv")

config <- config::get()

con <- DBI::dbConnect(
  RPostgres::Postgres(),
  host = config$db$host,
  port = config$db$port,
  dbname = config$db$database,
  user = config$db$user,
  password = config$db$password
)

stations <- DBI::dbGetQuery(con, "select * from stations") %>%
  as_tibble()

DBI::dbDisconnect(con)

x4 <- x2 %>%
  left_join(select(stations, station_id = id, name), by = c("station_id")) %>%
  select(station_id, name, stats) %>%
  unnest(stats) %>%
  mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out", "all-in", "all")))

x4 %>%
  write_csv("notes/20240410/stations-stats.csv")

x4 %>%
  ggplot(aes(value_cv, tau)) +
  geom_point() +
  geom_text_repel(aes(label = station_id), size = 2, alpha = 0.5) +
  facet_wrap(vars(split)) +
  labs(
    x = "CV[log10(obs. flow)]",
    y = "tau",
    title = "Tau vs Obs Flow Variability by Station",
    subtitle = "Higher flow variability -> higher tau values (except test-out)"
  ) +
  theme_bw() +
  theme(aspect.ratio = 1)
ggsave("notes/20240410/stations-tau-cv_flow.png", width = 8, height = 6)

x4 %>%
  ggplot(aes(score_s, tau)) +
  geom_point() +
  geom_text_repel(aes(label = station_id), size = 2, alpha = 0.5) +
  facet_wrap(vars(split)) +
  labs(
    x = "StDev[score]",
    y = "tau",
    title = "Tau vs Model Score Variability by Station",
    subtitle = "Higher variance of predicted scores -> higher tau values"
  ) +
  theme_bw() +
  theme(aspect.ratio = 1)
ggsave("notes/20240410/stations-tau-stdev_score.png", width = 8, height = 6)

x4 %>%
  ggplot(aes(value_cv, score_s)) +
  geom_point() +
  geom_text_repel(aes(label = station_id), size = 2, alpha = 0.5) +
  facet_wrap(vars(split)) +
  labs(
    x = "CV[log10(obs. flow)]",
    y = "StDev[score]",
    title = "Obs. Flow Variability vs Model Score Variability by Station",
    subtitle = "Higher flow variability -> higher score variability (except test-out)"
  ) +
  theme_bw() +
  theme(aspect.ratio = 1)
ggsave("notes/20240410/stations-cv_value-stdev_score.png", width = 8, height = 6)

x4 %>%
  arrange(station_id) %>%
  mutate(name = fct_inorder(glue("{station_id}-{name}"))) %>%
  ggplot(aes(name, tau)) +
  geom_col(position = position_dodge(), alpha = 0.5) +
  # scale_fill_brewer(palette = "Set1") +
  geom_text(aes(label = sprintf("%.2f", tau)), hjust = 1.1) +
  coord_flip() +
  ylim(0, 1) +
  facet_wrap(vars(split), nrow = 1) +
  labs(x = NULL, title = "Model Performance (Kendall Tau) by Station and Split") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
  )
ggsave("notes/20240410/stations-tau.png", width = 12, height = 4)

x %>%
  left_join(select(stations, station_id = id, name), by = "station_id") %>%
  mutate(name = glue("{station_id}-{name}")) %>%
  unnest(data) %>%
  mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out"))) %>%
  ggplot(aes(log10(pmax(value, 0.01)), score)) +
  geom_hex() +
  scale_fill_viridis_c(limits = c(1, NA), trans = "log10") +
  facet_grid(vars(split), vars(name)) +
  labs(
    x = "log10[obs. flow]", y = "score",
    title = "Obs. Flow vs Model Score by Image"
  ) +
  theme_bw() +
  theme(
    aspect.ratio = 1,
    strip.text.x = element_text(size = 5)
  )
ggsave("notes/20240410/images-value-score.png", width = 20, height = 7)

x %>%
  left_join(select(stations, station_id = id, name), by = "station_id") %>%
  mutate(name = glue("{station_id}-{name}")) %>%
  unnest(data) %>%
  filter(!is.na(value), !is.na(score)) %>%
  group_by(station_id) %>%
  mutate(
    n = n(),
    rank_value = (rank(value) - 1) / (n - 1),
    rank_score = (rank(score) - 1) / (n - 1)
  ) %>%
  mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out"))) %>%
  ggplot(aes(rank_value, rank_score)) +
  geom_hex() +
  geom_abline() +
  scale_fill_viridis_c(limits = c(1, NA), trans = "log10") +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent) +
  facet_grid(vars(split), vars(name)) +
  labs(
    x = "rank[obs. flow]", y = "rank[score]",
    title = "Rank(Obs. Flow) vs Rank(Model Score) by Image"
  ) +
  theme_bw() +
  theme(
    aspect.ratio = 1,
    strip.text.x = element_text(size = 5),
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
  )
ggsave("notes/20240410/images-rank_value-rank_score.png", width = 20, height = 7)


for (id in x2$station_id) {
  name <- stations$name[which(stations$id == id)]
  station_dir <- glue("notes/20240410/stations/{id}-{name}")
  dir.create(station_dir, showWarnings = FALSE)
  diagnostics_from <- glue("/mnt/d/fpe/rank/{id}/models/RANK-FLOW-20240410/RANK-FLOW-20240410.html")
  file.copy(diagnostics_from, station_dir, overwrite = TRUE)
  predictions_from <- glue("/mnt/d/fpe/rank/{id}/models/RANK-FLOW-20240410/transform/predictions.csv")
  file.copy(predictions_from, station_dir, overwrite = TRUE)

  p <- x %>%
    filter(station_id == id) %>%
    unnest(data) %>%
    filter(!is.na(value), !is.na(score)) %>%
    arrange(timestamp) %>%
    mutate(
      n = n(),
      rank_value = (rank(value) - 1) / (n - 1),
      rank_score = (rank(score) - 1) / (n - 1)
    ) %>%
    mutate(split = factor(split, levels = c("train", "val", "test-in", "test-out"))) %>%
    ggplot(aes(timestamp)) +
    geom_line(aes(y = rank_value, linetype = "Obs. Flow")) +
    # geom_line(aes(y = rank_score), color = "deepskyblue") +
    geom_point(aes(y = rank_score, color = split), size = 0.5, alpha = 0.5) +
    scale_color_brewer("Score", palette = "Set1") +
    # scale_x_continuous(labels = scales::percent) +
    scale_y_continuous(labels = scales::percent, expand = expansion(), limits = c(0, 1), breaks = scales::pretty_breaks(n = 8)) +
    # facet_grid(vars(split), vars(name)) +
    labs(
      linetype = NULL,
      y = "rank",
      title = "Rank(Obs. Flow) vs Rank(Model Score) by Image"
    ) +
    theme_bw() +
    theme(
    )
  ggsave(glue("notes/20240410/stations/{id}-{name}/ts-rank.png"), plot = p, width = 16, height = 4)
}
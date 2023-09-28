generate_report <- function (site, model_id, root_dir = "D:/fpe/sites") {
  output_file <- paste0("qmd/rank-", site, "-", model_id, ".html")

  quarto::quarto_render(
    input = "qmd/rank-predictions.qmd",
    output_file = output_file,
    execute_params = list(
      root_dir = root_dir,
      site = site,
      model_id = model_id
    )
  )
  target_filename <- file.path(root_dir, site, "models", model_id, basename(output_file))
  file.copy(output_file, target_filename, overwrite = TRUE)
  print(paste0("Report saved: ", target_filename))
}


generate_report("AVERYBB", "20230922")
generate_report("WESTB0", "20230922")
generate_report("LANESV", "20230922")
generate_report("VLYB", "20230922")
generate_report("PATRC", "20230922")



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

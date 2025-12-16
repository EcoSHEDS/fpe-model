library(tidyverse)

generate_report <- function (station_id, model_code, directory = "/mnt/d/fpe/rank/") {
  cat("generating report:", station_id, "\n")

  quarto::quarto_render(
    input = "qmd/rank-report.qmd",
    output_format = "html",
    execute_params = list(
      directory = directory,
      station_id = station_id,
      model_code = model_code
    ),
    debug = TRUE
  )
  target_filename <- file.path(
    directory, station_id, "models", model_code,
    paste0(model_code, ".html")
  )
  file.copy("qmd/rank-report.html", target_filename, overwrite = TRUE)
  print(paste0("Report saved: ", target_filename))
}

# generate_report(41, "RANK-FLOW-20251212", directory = "/mnt/d/fpe/rank/20251212/")

station_ids <- read_csv("/mnt/d/fpe/rank/20251212/stations.txt", col_names = "station_id")$station_id
walk(station_ids, \(x) generate_report(x, "RANK-FLOW-20251212", directory = "/mnt/d/fpe/rank/20251212/"))


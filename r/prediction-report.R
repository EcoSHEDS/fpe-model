generate_report <- function (site, root_dir = "D:/fpe/sites", filename = "predictions.csv") {
  output_file <- paste0("qmd/predictions-", site, ".html")

  quarto::quarto_render(
    input = "qmd/rank-predictions.qmd",
    output_file = output_file,
    execute_params = list(
      site = site,
      root_dir = root_dir,
      filename = filename
    )
  )
  file.copy(output_file, file.path(root_dir, site, "transform"), overwrite = TRUE)
  print(paste0("Report saved: ", file.path(root_dir, site, "transform", output_file)))
}


generate_report("AVERYBB")
generate_report("WESTB0")
generate_report("GREENR")
generate_report("WESTKILL")

sapply(c("AVERYBB", "WESTB0", "GREENR", "WESTKILL"), generate_report)

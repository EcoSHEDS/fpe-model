library(tidyverse)

sites <- c("AVERYBB", "WESTB0")

for (s in sites) {
  cat(s, "\n")
  x <- read_csv(file.path("../data", s, "flow-images.csv"), show_col_types = FALSE)
  x %>% 
    mutate(timestamp = with_tz(timestamp, "US/Eastern")) %>% 
    filter(
      hour(timestamp) %in% 7:18,
      month(timestamp) %in% 4:11
    ) %>% 
    arrange(timestamp) %>% 
    group_by(date = as_date(timestamp), hour = hour(timestamp)) %>% 
    slice(1) %>% 
    ungroup() %>% 
    select(-date, -hour) %>% 
    write_csv(file.path("../data", s, "flow-images-subset.csv"))
}

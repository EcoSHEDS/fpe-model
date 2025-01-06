setwd("~/git/fpe-model/notes/20241231-AVERYBB-aux")

library(tidyverse)
library(janitor)
library(glue)
library(daymetr)
library(slider)
library(yaml)

# avery brook bridge
coords <- c(42.44981, -72.69435)



# images ------------------------------------------------------------------

images <- read_csv("data/images.csv") %>% 
  mutate(
    timestamp = with_tz(timestamp, "America/New_York"),
    date = as_date(timestamp)
  ) %>% 
  filter(
    year(timestamp) <= 2023,
    !is.na(value)
  ) %>% 
  mutate(
    value = log10(value)
  ) %>% 
  select(-split)

images_train_val <- images %>%
  filter(date < ymd(20230601))
images_train <- images_train_val %>%
  sample_n(size = 1000) %>%
  mutate(split = "train") %>%
  select(split, image_id, filename, timestamp, date, value) %>%
  print()
images_val <- images_train_val %>%
  filter(!image_id %in% images_train$image_id) %>%
  sample_n(size = 200) %>%
  mutate(split = "val") %>%
  select(split, image_id, filename, timestamp, date, value) %>%
  print()
images_test_in <- images_train_val %>%
  filter(
    !image_id %in% images_train$image_id,
    !image_id %in% images_val$image_id
  ) %>%
  mutate(split = "test-in") %>%
  select(split, image_id, filename, timestamp, date, value)
images_test_out <- images %>%
  filter(date > max(images_train_val$date)) %>%
  mutate(split = "test-out") %>%
  select(split, image_id, filename, timestamp, date, value)

images_split <- bind_rows(
  images_train,
  images_val,
  images_test_in,
  images_test_out
)

images_split %>% 
  ggplot(aes(timestamp, value)) +
  geom_point(aes(color = split), size = 0.5)

tabyl(images_split, split)

images_split %>%
  mutate(
    across(starts_with("timestamp"), ~ format(with_tz(., "UTC"), "%Y-%m-%d %H:%M:%S"))
  ) %>% 
  write_csv("data/images-split.csv")

# aux: daymet ------------------------------------------------------------------

aux_daymet_raw <- daymetr::download_daymet(
  site = "AVERYBB",
  lat = coords[[1]],
  lon = coords[[2]],
  start = min(year(images$timestamp)),
  end = 2023,
)

aux_daymet <- aux_daymet_raw$data %>% 
  clean_names() %>% 
  as_tibble() %>% 
  mutate(
    date = ymd(glue("{year}-01-01")) + days(yday - 1)
  ) %>% 
  complete(date = seq.Date(from = min(date), to = max(date), by = "day")) %>% 
  transmute(
    date,
    dayl = dayl_s,
    prcp = prcp_mm_day,
    srad = srad_w_m_2,
    swe = swe_kg_m_2,
    tmax = tmax_deg_c,
    tmin = tmin_deg_c,
    vp = vp_pa
  ) %>%
  mutate(
    across(
      -date,
      \(x) scale(x)[,1], .names = "{.col}_z"
    )
  ) %>%
  print()


aux_daymet %>%
  write_csv("data/aux.csv")


# run 01-resnet ------------------------------------------------------------------

dir.create("runs/01-resnet/data", showWarnings = FALSE, recursive = TRUE)
file.copy("data/images-split.csv", "runs/01-resnet/data/images.csv", overwrite = TRUE)


x_1 <- read_csv("runs/01-resnet/output/data/predictions.csv") %>% 
  arrange(timestamp)


# run 02-lstm -------------------------------------------------------------

dir.create("runs/02-lstm/data", showWarnings = FALSE, recursive = TRUE)
file.copy("data/images-split.csv", "runs/02-lstm/data/labels.csv", overwrite = TRUE)
file.copy("data/aux.csv", "runs/02-lstm/data/aux.csv", overwrite = TRUE)

x_2 <- read_csv("runs/02-lstm/output/data/02-lstm-predictions.csv") %>% 
  arrange(timestamp)

x_2 %>% 
  ggplot(aes(value, prediction)) +
  geom_point(size = 0.1)

x_2 %>% 
  ggplot(aes(rank(value), rank(prediction))) +
  geom_point(size = 0.1)

x_2 %>% 
  ggplot(aes(timestamp)) +
  geom_line(aes(y = value, color = "obs")) +
  geom_line(aes(y = prediction, color = "pred"))

bind_rows(
  resnet = x_1,
  lstm = x_2,
  .id = "model"
) %>% 
  ggplot(aes(value, prediction, color = split)) +
  geom_point(size = 0.1) +
  facet_wrap(vars(model))


# tau vs n images ---------------------------------------------------------

x <- read_csv("~/data/fpe/experiments/20241111-WB0-n_train/runs/n_train_1000/output/test/predictions.csv")

x_tau <- tibble(
  n = c(seq(10, 90, by = 10), seq(100, 16000, by = 100))
) %>% 
  mutate(
    data = map(n, \(n) sample_n(x, size = n)),
    tau = map_dbl(data, \(x) cor(x$value, x$score, method = "kendall"))
  )
x_tau %>% 
  ggplot(aes(n, tau)) +
  geom_point()


# run 03-resnet-lstm ---------------------------------------------------------


dir.create("runs/03-resnet-lstm/data", showWarnings = FALSE, recursive = TRUE)
file.copy("data/images-split.csv", "runs/03-resnet-lstm/data/images.csv", overwrite = TRUE)
file.copy("data/aux.csv", "runs/03-resnet-lstm/data/aux.csv", overwrite = TRUE)


x_3 <- read_csv("runs/03-resnet-lstm/output/data/03-resnet-lstm-predictions.csv") %>% 
  arrange(timestamp)


bind_rows(
  resnet = x_1,
  lstm = x_2,
  .id = "model"
) %>% 
  group_by(model, date = as_date(timestamp)) %>% 
  summarise(across(c(value, prediction), mean)) %>% 
  ggplot(aes(value, prediction)) +
  geom_point(size = 0.1) +
  facet_wrap(vars(model))




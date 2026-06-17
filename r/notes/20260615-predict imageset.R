# confirm the new predict imageset container generates the same predictions
# as the original transform script

library(tidyverse)
library(glue)
library(paws)

STATION_ID <- 29
MODEL_CODE <- "RANK-FLOW-20240410"
IMAGESET_UUID <- "5a53b364-7a42-4708-b66e-d837c6b05f3e"

S3_STORAGE_BUCKET <- "usgs-chs-conte-prod-fpe-storage"
S3_MODEL_BUCKET <- "usgs-chs-conte-prod-fpe-models"


Sys.setenv(AWS_PROFILE = "conte-prod")
s3 <- paws::s3()
s3$list_buckets()

# load: predict imageset ----
pred_sm_key <- glue::glue("rank/{STATION_ID}/models/{MODEL_CODE}/transform/imagesets/{IMAGESET_UUID}/predictions.csv")
pred_sm_obj <- s3$get_object(
  Bucket = S3_MODEL_BUCKET,
  Key = pred_sm_key
)

pred_sm_csv <- rawToChar(pred_sm_obj$Body)
pred_sm <- read_csv(pred_sm_csv)


# load: transform ----
pred_tr_key <- glue::glue("rank/{STATION_ID}/models/{MODEL_CODE}/transform/predictions.csv")
pred_tr_obj <- s3$get_object(
  Bucket = S3_MODEL_BUCKET,
  Key = pred_tr_key
)

# Convert raw connection to text, then parse CSV
pred_tr_csv <- rawToChar(pred_tr_obj$Body)
pred_tr <- read_csv(pred_tr_csv)

# compare ----

df <- bind_rows(
  sm = pred_sm,
  tr = pred_tr %>% 
    filter(image_id %in% pred_sm$image_id),
  .id = "source"
)

df %>% 
  select(source, image_id, score) %>% 
  pivot_wider(names_from = "source", values_from = "score") %>% 
  mutate(diff = tr - sm) %>% 
  summary()

df %>% 
  select(source, image_id, score) %>% 
  pivot_wider(names_from = "source", values_from = "score") %>% 
  ggplot(aes(sm, tr)) +
  geom_abline() +
  geom_point()

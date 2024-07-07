library(tidyverse)
library(nasapower)

# https://doi.org/10.31220/agriRxiv.2024.00251
tab_coord <- data.frame(
  location = c("Scottsbluff", "Lincoln", "MOValley", "Ames", "Crawfordsville"),
  lat = c(41.85, 40.86, 41.67, 42.01, 41.19),
  lon = c(-103.70, -96.61, -95.94, -93.73, -91.48)
)

dataset <- "train"  # train or validation
year <- 2022  # 2022 or 2023
files <- list(
  train2022 = "HYBRID_HIPS_V3.5_ALLPLOTS.csv",
  train2023 = "train_HIPS_HYBRIDS_2023_V2.3.csv",
  validation2023 = "val_HIPS_HYBRIDS_2023_V2.3.csv"
)
filename <- paste0(
  "data/", dataset, "/", year, 
  "/DataPublication_final/GroundTruth/", files[paste0(dataset, year)]
)
if (dataset == "validation") {
  filename <- gsub("/DataPublication_final", "", filename)
}

# field data
tab_field <- read.csv(filename) %>% 
  mutate(plantingDate = ifelse(
    dataset == "train" & year == 2022, 
    as.Date(plantingDate), 
    as.Date(plantingDate, "%m/%d/%y")
  )) %>% 
  mutate(plantingDate = as.Date(plantingDate)) %>% 
  select(location, plantingDate) %>% 
  unique() %>% 
  arrange(location, plantingDate) %>% 
  group_by(location) %>% 
  filter(plantingDate == min(plantingDate)) %>% 
  ungroup() %>% 
  left_join(tab_coord, by = "location") %>% 
  rename(w0 = plantingDate) %>% 
  select(location, lat, lon, w0) %>% 
  as.data.frame()

weeks <- 14
for (i in 1:weeks) {
  tab_field[, paste0("w", i)] <- tab_field$w0 + (i * 7)
}

pars <- c(
  "T2M",  # C
  "PRECTOTCORR"  # mm/day
)

tab_climate <- data.frame()
for (i in 1:nrow(tab_field)) {
  location <- tab_field[i, "location"]
  tab_weeks <- data.frame()
  for (j in 1:weeks) {
    t1 <- tab_field[i, paste0("w", j - 1)]
    t2 <- tab_field[i, paste0("w", j)]
    tab_power <- get_power(
      community = "ag",
      lonlat = unlist(tab_field[i, c("lon", "lat")]),
      pars = pars,
      dates = c(t1, t2),
      temporal_api = "daily"
    )[, pars]
    tab_power_agg <- tab_power %>% 
      summarise_all(
        list(
          # min = min, 
          q1 = function(x) quantile(x, 0.25), 
          median = median, 
          q3 = function(x) quantile(x, 0.75), 
          max = max,
          mean = mean, 
          sd = sd
        )
      ) %>% 
      mutate(location = location) %>% 
      mutate(period = paste0("w", j - 1, "-w", j))
    tab_weeks <- bind_rows(tab_weeks, tab_power_agg)
  }
  tab_climate <- bind_rows(tab_climate, tab_weeks)
  cat("location:", location, "\n")
}

tab_climate_wider <- tab_climate %>% 
  pivot_wider(names_from = period, values_from = -c(location, period))

# output
outname <- paste0("output/climate_", dataset, "_", year, ".csv")
write.csv(tab_climate_wider, outname, row.names = F)


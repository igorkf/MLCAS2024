library(tidyverse)

plot_series <- function(tab, vi) {
  tab %>% 
    group_by(environment, tp) %>% 
    summarise(
      mean = cor(yieldPerAcre, !!sym(paste0(vi, "_mean"))),
      median = cor(yieldPerAcre, !!sym(paste0(vi, "_median"))),
      min = cor(yieldPerAcre, !!sym(paste0(vi, "_min"))),
      max = cor(yieldPerAcre, !!sym(paste0(vi, "_max")))
    ) %>% 
    pivot_longer(-c(environment, tp)) %>% 
    ggplot(aes(x = tp, y = value, color = name, group = name)) +
    geom_point() +
    geom_line() + 
    facet_wrap(~environment) +
    labs(title = vi)
}

train <- read.csv("output/train.csv")
train_sat <- read.csv("output/satellite_train_2022.csv") %>% 
  mutate(img_id = paste0(location, "_", experiment, "_", range, "_", row)) %>% 
  left_join(train[, c("img_id", "yieldPerAcre")], by = "img_id") %>% 
  filter(!is.na(yieldPerAcre)) %>% 
  mutate_at(vars(location, tp), as.factor) %>% 
  mutate(environment = paste0(location, "_2022")) %>% 
  mutate(year = "2022") %>% 
  select(-c(location, experiment, range, row, path, file, img_id))

val <- read.csv("output/val.csv")
val_sat <- read.csv("output/satellite_train_2023.csv") %>% 
  mutate(img_id = paste0(location, "_", experiment, "_", range, "_", row)) %>% 
  left_join(val[, c("img_id", "yieldPerAcre")], by = "img_id") %>% 
  filter(!is.na(yieldPerAcre)) %>% 
  mutate_at(vars(location, tp), as.factor) %>% 
  mutate(environment = paste0(location, "_2023")) %>% 
  mutate(year = "2023") %>% 
  select(-c(location, experiment, range, row, path, file, img_id))

sat <- bind_rows(train_sat, val_sat)

vis <- c(
  "NDVI",
  "NDRE",
  "EVI",
  "NGRDI",
  "GNDVI",
  "GLI"
) 

# correlation by year
sat_corr <- tibble()
cols_vis <- colnames(select(sat, contains(vis)))
for (i in 1:length(cols_vis)) {
  vi <- cols_vis[i]
  corr <- sat %>% 
    group_by(year, tp) %>% 
    summarise(corr = cor(yieldPerAcre, !!sym(vi))) %>% 
    mutate(vi = vi)
  sat_corr <- bind_rows(sat_corr, corr)
}

# correlation by environment
sat_corr_env <- tibble()
for (i in 1:length(cols_vis)) {
  vi <- cols_vis[i]
  corr <- sat %>% 
    group_by(environment, tp) %>% 
    summarise(corr = cor(yieldPerAcre, !!sym(vi))) %>% 
    mutate(vi = vi)
  sat_corr_env <- bind_rows(sat_corr_env, corr)
}

# sat_corr_wider <- sat_corr %>% 
#   pivot_wider(names_from = year, values_from = corr, names_prefix = "vi_") %>% 
#   mutate(diff = abs(vi_2022 - vi_2023))

sat_corr_env %>% 
  filter(str_detect(vi, "max") == T) %>% 
  ggplot(aes(x = tp, y = corr, color = vi, group = vi)) +
    geom_line() +
    geom_point() + 
    facet_wrap(~environment)

plot_series(sat, "NDVI")
plot_series(sat, "NDRE")
plot_series(sat, "EVI")
plot_series(train_sat, "NGRDI")
plot_series(train_sat, "GNDVI")
plot_series(train_sat, "GLI")
plot_series(train_sat, "SAVI")
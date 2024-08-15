library(tidyverse)
library(readxl)

read_field <- function(path) {
  tab <- read.csv(path) %>% 
    filter(!is.na(plantingDate)) %>%
    dplyr::select(location, plantingDate) %>% 
    unique() %>% 
    group_by(location) %>% 
    filter(plantingDate == min(plantingDate)) %>%  # because Ames has 2 plantingDates (2022-05-22, 2022-05-23)
    ungroup()
  return(tab)
}

read_date <- function(path, tab_field) {
  tab <- read_excel(path) %>% 
    rename(location = Location) %>% 
    filter(Image == "Satellite") %>% 
    filter(location != "North Platte") %>% 
    mutate(location = str_replace_all(location, "Missouri Valley", "MOValley")) %>% 
    dplyr::select(-Image) %>% 
    pivot_wider(names_from = time, values_from = Date) %>% 
    inner_join(tab_field) %>% 
    mutate(sat_1st = as.integer(TP1 - plantingDate)) %>% 
    mutate(sat_2nd = as.integer(TP2 - plantingDate)) %>% 
    mutate(sat_3rd = as.integer(TP3 - plantingDate)) %>% 
    mutate(sat_4th = as.integer(TP4 - plantingDate))
  return(tab)
}

field_2022 <- read_field("data/train/2022/DataPublication_final/GroundTruth/HYBRID_HIPS_V3.5_ALLPLOTS.csv") %>% 
  mutate(plantingDate = as.POSIXct(plantingDate))
date_2022 <- read_date("data/train/2022/DataPublication_final/GroundTruth/DateofCollection.xlsx", field_2022)

field_2023 <- read_field("data/train/2023/DataPublication_final/GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv") %>% 
  mutate(plantingDate = as.POSIXct(as.Date(plantingDate, "%m/%d/%y")))
date_2023 <- read_date("data/train/2023/DataPublication_final/GroundTruth/DateofCollection.xlsx", field_2023)

field_val_2023 <- read_field("data/validation/2023/GroundTruth/val_HIPS_HYBRIDS_2023_V2.3.csv") %>% 
  mutate(plantingDate = as.POSIXct(as.Date(plantingDate, "%m/%d/%y")))
date_val_2023 <- read_date("data/validation/DateofCollection.xlsx", field_val_2023)

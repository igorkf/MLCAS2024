library(tidyverse)

tabs1 <- readxl::read_excel("data/TPJ_16123_TableS1.xlsx") %>% 
  select(ID, Group)

field2022 <- read.csv("data/train/2022/DataPublication_final/GroundTruth/HYBRID_HIPS_V3.5_ALLPLOTS.csv") %>% 
  select(genotype, yieldPerAcre)
field2023 <- read.csv("data/train/2023/DataPublication_final/GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv") %>% 
  select(genotype, yieldPerAcre)
field2023_val <- read.csv("data/validation/2023/GroundTruth/val_HIPS_HYBRIDS_2023_V2.3.csv") %>% 
  select(genotype, yieldPerAcre)

# just to fill non-genotyped hybrids later in the kinship matrix
all_parents <- bind_rows(field2022, field2023, field2023_val) %>%
  select(-yieldPerAcre) %>%
  unique() %>% 
  filter(!is.na(genotype))
write.csv(all_parents, "output/all_parents.csv", row.names = F)

parents <- bind_rows(field2022, field2023, field2023_val) %>% 
  select(-yieldPerAcre) %>% 
  mutate(parent1 = str_split_i(genotype, " X ", 1)) %>% 
  mutate(parent2 = str_split_i(genotype, " X ", 2))

genos <- bind_rows(
  tibble(genotype = unique(parents$parent1)),
  tibble(genotype = unique(parents$parent2))
) %>%
  unique() %>% 
  mutate(genotype_fix = case_when(
    genotype == "C.I. 540" ~ "CI_540",
    genotype == "L 289" ~ "L_289",
    genotype == "OS426" ~ "Os426",
    genotype == "MO17" ~ "Mo17",
    genotype == "3IIH6" ~ "DK3IIH6",
    genotype == "OH43" ~ "Oh43",
    genotype == "CI 3A" ~ "CI_3A",
    genotype == "WF9" ~ "Wf9",
    genotype == "'IOWA I 205'" ~ "I_205",
    TRUE ~ as.character(genotype)
  )) %>% 
  left_join(tabs1, by = c("genotype_fix" = "ID")) %>% 
  filter(!is.na(genotype))

colSums(is.na(genos))
colSums(is.na(genos)) / nrow(genos)

# write mapping
genos %>%
  filter(!is.na(Group)) %>% 
  write.csv("output/geno_mapping.csv", row.names = F)

# write list
genos %>% 
  filter(!is.na(Group)) %>% 
  select(genotype_fix) %>% 
  write.table(
    "output/geno_samples.txt", 
    row.names = F, col.names = F, quote = F
  )
  
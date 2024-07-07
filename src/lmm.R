library(desplot)
library(dplyr)
library(asreml)

train2022 <- read.csv("data/train/2022/DataPublication_final/GroundTruth/HYBRID_HIPS_V3.5_ALLPLOTS.csv") %>% 
  filter(!is.na(yieldPerAcre)) %>% 
  mutate_at(vars(location, experiment, block, range, row), factor)
train2023 <- read.csv("data/train/2023/DataPublication_final/GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv") %>% 
  mutate_at(vars(location, experiment, block, range, row), factor)
train <- bind_rows(train2022, train2023) %>% 
  mutate_at(vars(genotype, irrigationProvided, nitrogenTreatment, poundsOfNitrogenPerAcre), factor)


with(train, table(experiment, location))

# linear mixed model
mod1 <- asreml(
  fixed = yieldPerAcre ~ location + experiment + experiment:block,
  random = ~ genotype,
  data = train
)
wald.asreml(mod1)
plot(mod1)

# BLUPs
blups <- summary(mod, coef = T)$coef.random
hist(blups[, "solution"])
write.csv(as.data.frame(blups), "output/blups.csv", row.names = T)

# table(train2022$experiment)
# table(train2022$genotype)
# with(train2022, table(genotype, experiment))
# with(train2022, table(block, experiment))
# with(train2022, table(genotype, block))
# with(train2022, table(row, range))

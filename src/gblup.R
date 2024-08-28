library(tidyverse)
library(sommer)

source("src/utils.R")

val <- read.csv("output/val.csv")
val$dataset <- "val"
val$sqrt_y <- sqrt(val$yieldPerAcre)
val$two_thirds_y <- val$yieldPerAcre ^ (2 / 3)
val$three_quarters_y <- val$yieldPerAcre ^ (3 / 4)
val$log_y <- log(val$yieldPerAcre)
val$env <- paste0(val$location, "_", val$year)
val$env_exp <- paste0(val$env, "_", val$experiment)

train <- read.csv("output/train.csv")
train$dataset <- "train"
train$sqrt_y <- sqrt(train$yieldPerAcre)
train$two_thirds_y <- train$yieldPerAcre ^ (2 / 3)
train$three_quarters_y <- train$yieldPerAcre ^ (3 / 4)
train$log_y <- log(train$yieldPerAcre)
train$env <- paste0(train$location, "_", train$year)
train$env_exp <- paste0(train$env, "_", train$experiment)

test <- read.csv("output/test.csv")
test$yieldPerAcre <- NA
test$dataset <- "test"
test$yieldPerAcre <- 1
test$sqrt_y <- sqrt(test$yieldPerAcre)
test$two_thirds_y <- test$yieldPerAcre ^ (2 / 3)
test$three_quarters_y <- test$yieldPerAcre ^ (3 / 4)
test$log_y <- log(test$yieldPerAcre)
test$env <- paste0(test$location, "_", test$year)
test$env_exp <- paste0(test$env, "_", test$experiment)

# summary
summary(train)

# process commercial hybrids
train <- train %>% 
  mutate(commercial = ifelse(grepl(" X ", genotype), "no", "yes")) %>% 
  mutate(genotype = ifelse(commercial == "yes", paste0(genotype, " X void"), genotype))
val <- val %>% 
  mutate(commercial = ifelse(grepl(" X ", genotype), "no", "yes")) %>% 
  mutate(genotype = ifelse(commercial == "yes", paste0(genotype, " X void"), genotype))
test <- test %>% 
  mutate(commercial = ifelse(grepl(" X ", genotype), "no", "yes")) %>% 
  mutate(genotype = ifelse(commercial == "yes", paste0(genotype, " X void"), genotype))

train[c("parent1", "parent2")] <- str_split_fixed(train$genotype, " X ", 2)
val[c("parent1", "parent2")] <- str_split_fixed(val$genotype, " X ", 2)
test[c("parent1", "parent2")] <- str_split_fixed(test$genotype, " X ", 2)

cats <- c("experiment", "img_id", "genotype", "commercial", "nitrogenTreatment",
          "year", "parent1", "parent2", "env", "env_exp")
for (cat in cats) {
  train[, cat] <- as.factor(train[, cat])
  val[, cat] <- as.factor(val[, cat])  
  test[, cat] <- as.factor(test[, cat])  
}

# vegetation indexes
vis <- c(
  colnames(train)[grep("^NDVI_", colnames(train))],
  colnames(train)[grep("^NDRE_", colnames(train))]
  # colnames(train)[grep("^MTCI_", colnames(train))]
  # colnames(train)[grep("^CI_", colnames(train))]
  # colnames(train)[grep("NGRDI", colnames(train))],
  # colnames(train)[grep("GNDVI", colnames(train))],
  # colnames(train)[grep("GLI", colnames(train))]
)
cat("VIs:", vis, "\n")
cat("# VIs:", length(vis), "\n")

# enviromic relationship kinship
all_data <- bind_rows(train, val, test)
V <- all_data %>%
  dplyr::select(img_id, NDVI_mean_fixed_2, NDVI_median_fixed_2, NDVI_min_fixed_2,
                NDRE_mean_fixed_2, NDRE_max_fixed_2) %>% 
  as.matrix()
rownames(V) <- V[, 1]
V <- V[, -1]
class(V) <- "numeric"
V <- apply(V, 2, function(x) (x - mean(x)))
K <- tcrossprod(V) / mean(diag(V))
# K <- arc(K)
dim(K)
# heatmap(K)
# K_melted <- reshape::melt(W)
# ggplot(head(K_melted, 100000), aes(x = X1, y = X2, fill = value)) +
#   geom_tile()
# hist(diag(K))

# GRMs
G <- as.matrix(read.table("output/G.txt"))
colnames(G) <- rownames(G)  # fix colnames
G1 <- G2 <- G

# G1 <- arc(G1)
unknown_p1 <- setdiff(levels(all_data$parent1), rownames(G))
G1 <- bind_block_diag(G1, unknown_p1)

# G2 <- arc(G2)
unknown_p2 <- setdiff(levels(all_data$parent2), rownames(G))
G2 <- bind_block_diag(G2, unknown_p2)

G1G2 <- as.matrix(read.table("output/G1G2.txt"))
# G1G2 <- arc(G1G2)
colnames(G1G2) <- rownames(G1G2)  # fix colnames
unknown_geno <- setdiff(levels(all_data$genotype), rownames(G1G2))
Ggeno <- as.matrix(Matrix::bdiag(G1G2, diag(length(unknown_geno))))
rownames(Ggeno) <- colnames(Ggeno) <- c(rownames(G1G2), unknown_geno)

# rownames(G)[rownames(G) == " "] <- "void"
# colnames(G)[colnames(G) == " "] <- "void"
# Garc <- as.matrix(bWGR::EigenARC(G))
# rownames(Garc) <- colnames(Garc) <- rownames(G)
# G <- Garc
# G[47:56, 47:56] <- diag(10) # put diagonal block on commercials
# dim(G)
# heatmap(G)
# G1 <- G[rownames(G) %in% levels(all_data$parent1), rownames(G) %in% levels(all_data$parent1)]
# dim(G1)
# G2 <- G[rownames(G) %in% levels(all_data$parent2), rownames(G) %in% levels(all_data$parent2)]
# dim(G2)
# GG <- kronecker(G1, G2, make.dimnames = T)
# rownames(GG) <- colnames(GG) <- sub(":", " X ", rownames(GG))
# GG <- GG[rownames(GG) %in% unique(all_data$genotype), rownames(GG) %in% unique(all_data$genotype)]
# dim(GG)
# GG[1:5, 1:6]
# rownames(GG)[grepl("void", rownames(GG))]

# add some covariance due within same company
# hist(GG[upper.tri(GG, diag = F)])
# hoeg1 <- which(rownames(GG) == "HOEGEMEYER 7089 AMXT X void")
# hoeg2 <- which(rownames(GG) == "HOEGEMEYER 8065RR X void")
# GG[hoeg1, hoeg2] <- GG[hoeg2, hoeg1] <- 1
# pio1 <- which(rownames(GG) == "PIONEER 1311 AMXT X void")
# pio2 <- which(rownames(GG) == "PIONEER P0589 AMXT X void")
# GG[pio1, pio2] <- GG[pio2, pio1] <- 1
# syn1 <- which(rownames(GG) == "SYNGENTA NK0760-3111 X void")
# syn2 <- which(rownames(GG) == "SYNGENTA NK0659-3120-EZ1 X void")
# GG[syn1, syn2] <- GG[syn2, syn1] <- 1

# contingency tables
with(all_data, table(env, irrigationProvided))  # only Scottsbluff_2022 was irrigated
with(all_data, table(env, nitrogenTreatment))  # kinda balanced
with(all_data, table(env, poundsOfNitrogenPerAcre))
with(all_data, table(env, plotLength))  # only Scottsbluff_2022 used 22.5

# mixed models
# gxe models: https://cran.r-project.org/web/packages/sommer/vignettes/v4.sommer.gxe.pdf
fixed <- as.formula(yieldPerAcre ~ NDVI_mean_fixed_2 + NDVI_median_fixed_2 + NDVI_min_fixed_2 + 
                    NDVI_sum_fixed_2 + NDRE_mean_fixed_2 + NDRE_max_fixed_2 + NDRE_sum_fixed_2)

# commercial + VIs + genotype
system.time(
  mod1 <- mmer(fixed, random = ~ vsr(parent1, Gu = G1) + vsr(parent2, Gu = G2) + vsr(img_id, Gu = K), # , + vsr(genotype, Gu = Ggeno),
               rcov= ~ units, data = train)
)
summary(mod1)
yhat1 <- build_prediction(
  mod1, fixed, val, "yieldPerAcre", add_p1 = F, add_p2 = F, add_env = T, add_geno = F, verbose = T
)
cat("RMSE:", RMSE(val$yieldPerAcre, yhat1), "\n")
cat("r:", cor(val$yieldPerAcre, yhat1), "\n\n")
train %>% 
  mutate(yhat = mod1$fitted) %>% 
    ggplot(aes(x = yieldPerAcre, y = yhat, color = env_exp)) +
    geom_point()

# commercial + nitrogenTreatment + VIs + p1 + p2
# fixed2 <- update(fixed, ~ nitrogenTreatment + ., data = all_data)
# system.time(
#   mod2 <- mmer(fixed2, random = ~ vsr(parent1, Gu = G1) + vsr(parent2, Gu = G2),
#                rcov = ~ units, data = train)
# )
# summary(mod2)
# yhat2 <- build_prediction(
#   mod2, fixed2, val, "yieldPerAcre", add_p1 = T, add_p2 = T, add_geno = F, add_env = F, verbose = T
# )
# cat("RMSE:", RMSE(val$yieldPerAcre, yhat2), "\n")
# cat("r:", cor(val$yieldPerAcre, yhat2), "\n\n")
# train %>% 
#   mutate(yhat = mod2$fitted) %>% 
#   ggplot(aes(x = yieldPerAcre, y = yhat, color = env_exp)) +
#   geom_point()

# commercial + nitrogenTreatment + VIs + genotype
# system.time(
#   mod3 <- mmer(fixed2,
#                random = ~ vsr(parent1, Gu = G1) + vsr(parent2, Gu = G2) + vsr(genotype, Gu = Ggeno),
#                rcov = ~ units, data = train)
# )
# summary(mod3)
# yhat3 <- build_prediction(
#   mod3, fixed2, val, "three_quarters_y", add_p1 = T, add_p2 = T, add_geno = T, add_env = F, verbose = T
# )
# cat("RMSE:", RMSE(val$yieldPerAcre, yhat3), "\n")
# cat("r:", cor(val$yieldPerAcre, yhat3), "\n\n")

# refit with full data
full <- droplevels(bind_rows(train, val))
system.time(
  mod_full <- mmer(fixed, random = ~ vsr(genotype, Gu = Ggeno), # vsr(parent1, Gu = G1) + vsr(parent2, Gu = G2),
                   rcov = ~ units, data = full)
)
summary(mod_full)
ypred <- build_prediction(
  mod_full, fixed, test, "yieldPerAcre", add_p1 = F, add_p2 = F, add_env = F, add_geno = T, verbose = T
)

# overlapping between 2022 and 2023
cat("overlapping of parent1 and parent2 among 2022 and 2023\n")
union_p1 <- unique(c(train$parent1, val$parent1))
inter_p1 <- intersect(train$parent1, val$parent1)
cat("p1:", length(inter_p1), "overlapping from total of", length(union_p1), "\n")
union_p2 <- unique(c(train$parent2, val$parent2))
inter_p2 <- intersect(train$parent2, val$parent2)
cat("p2:", length(inter_p2), "overlapping from total of", length(union_p2), "\n")

# check estimates
df_coef <- cbind(mod1$Beta[, 2:3], mod_full$Beta[, 3])
df_coef

# predict on sub
tab_sub <- read.csv("data/validation/2023/GroundTruth/val_HIPS_HYBRIDS_2023_V2.3.csv")
stopifnot(all(test$experiment == tab_sub$experiment))
tab_sub$yieldPerAcre <- ypred

# compare distributions
dists <- rbind(summary(train$yieldPerAcre), summary(val$yieldPerAcre), summary(tab_sub$yieldPerAcre))
rownames(dists) <- c("2022", "2023", "sub")
dists
write.csv(tab_sub, "output/submission.csv",row.names = F)

library(tidyverse)
library(lme4)

RMSE <- function(y, ypred) {
  sqrt((sum((y - ypred) ^ 2) / length(y)))
}

train <- read.csv("output/train.csv")
train[c("parent1", "parent2")] <- str_split_fixed(train$genotype, " X ", 2)
train$env <- interaction(train$location, train$year, sep = "_")
train$C <- train$range
train$R <- train$row

val <- read.csv("output/val.csv")
val[c("parent1", "parent2")] <- str_split_fixed(val$genotype, " X ", 2)
val$env <- interaction(val$location, val$year, sep = "_")
val$C <- val$range
val$R <- val$row

test <- read.csv("output/test.csv")
test[c("parent1", "parent2")] <- str_split_fixed(test$genotype, " X ", 2)
test$env <- interaction(test$location, test$year, sep = "_")
test$C <- test$range
test$R <- test$row

cats <- c("experiment", "img_id", "genotype", "parent1", "parent2", "location", "env", "C", "R", "nitrogenTreatment")
for (cat in cats) {
  train[, cat] <- as.factor(train[, cat])
  val[, cat] <- as.factor(val[, cat])  
  test[, cat] <- as.factor(test[, cat])  
}

# first model
vis <- c(
  colnames(train)[grep("^NDVI_", colnames(train))],
  colnames(train)[grep("^NDRE_", colnames(train))]
  # colnames(train)[grep("^MTMCI", colnames(train))],
  # colnames(train)[grep("^CI_", colnames(train))],
)
fixed_eff <- c(
  # "commercial",
  # "location",
  # "nitrogenTreatment",
  vis
)

# ggplot(train, aes(x = NDVI_mean_fixed, y = yieldPerAcre, color = location)) +
#   geom_point()
# ggplot(train, aes(x = NDVI_mean_fixed, y = sqrt(yieldPerAcre), color = location)) +
#   geom_point()


form <- formula(paste0("yieldPerAcre ~ ", paste0(fixed_eff, collapse = " + ")))
mod <- lm(form, data = train)
summary(mod)
val$yhat_lm <- predict(mod, newdata = val)
cat("linear model:\n")
cat("RMSE:", RMSE(val$yieldPerAcre, val$yhat_lm), "\n")
cat("r:", cor(val$yieldPerAcre, val$yhat_lm), "\n")
tab_res <- train %>% 
  mutate(yhat = mod$fitted.values) %>% 
  mutate(res = mod$residuals)
  
# plots
ggplot(tab_res, aes(x = yhat, y = res, color = location)) +
  geom_point()
ggplot(tab_res, aes(x = yieldPerAcre, y = yhat, color = location)) +
  geom_point()

# mixed effects model
form_mm <- formula(
  paste0(
    "yieldPerAcre ~ ", 
    paste0(fixed_eff, collapse = " + "),
    " + (1 | parent1) + (1 | parent2)"
  )
)
mod2 <- lmer(form_mm, data = train)
summary(mod2)
val$yhat_mm <- predict(mod2, newdata = val, allow.new.levels = T)
cat("call full model:\n")
summary(mod2)$call
cat("RMSE:", RMSE(val$yieldPerAcre, val$yhat_mm), "\n")
cat("r:", cor(val$yieldPerAcre, val$yhat_mm), "\n")

# all possible combinations
combinations <- unlist(lapply(1:length(fixed_eff), function(n) {
  combn(fixed_eff, n, simplify = FALSE)
}), recursive = FALSE)
best_rmse <- 1000000
for (vars in combinations) {
  form_mm <- formula(
    paste0(
      "yieldPerAcre ~ ", 
      paste0(vars, collapse = " + "),
      " + (1 | parent1) + (1 | parent2)"
    )
  )
  mod2 <- lmer(form_mm, data = train)
  yhat_mm <- predict(mod2, newdata = val, allow.new.levels = T)
  rmse <- RMSE(val$yieldPerAcre, yhat_mm)
  if (rmse < best_rmse) {
    mod_best <- mod2
    best_rmse <- rmse
    cat(vars, "\n")
    cat("RMSE:", rmse, "\n")
    cat("r:", cor(val$yieldPerAcre, yhat_mm), "\n\n")
  }
}
tab_res <- train %>% 
  mutate(yhat = fitted(mod_best)) %>% 
  mutate(res = resid(mod_best))

# plots
ggplot(tab_res, aes(x = yhat, y = res, color = location)) +
  geom_point()
ggplot(tab_res, aes(x = yieldPerAcre, y = yhat , color = location)) +
  geom_point()


# overlapping between 2022 and 2023
cat("overlapping of parent1 and parent2 among 2022 and 2023\n")
union_p1 <- unique(c(train$parent1, val$parent1))
inter_p1 <- intersect(train$parent1, val$parent1)
cat(length(inter_p1), "overlapping from total of", length(union_p1), "\n")
union_p2 <- unique(c(train$parent2, val$parent2))
inter_p2 <- intersect(train$parent2, val$parent2)
cat(length(inter_p2), "overlapping from total of", length(union_p2), "\n")

# refit with 2022 + 2023
val$yhat_lm <- NULL
val$yhat_mm <- NULL
full <- rbind(train, val)
mod_full <- update(mod_best, data = full)
cat("call full model:\n")
summary(mod_full)$call
cat("\n")
                          
# predict on sub
tab_sub <- read.csv("data/test/Test/GroundTruth/test_HIPS_HYBRIDS_2023_V2.3.csv")
stopifnot(all(test$experiment == tab_sub$experiment))
pred <- predict(mod_full, newdata = test, allow.new.levels = T)  # just one line unknown (ND203)

# compare estimates
df_coef <- data.frame(
  m_2022 = summary(mod_best)$coefficients[, "Estimate"],
  m_2022_2023 = summary(mod_full)$coefficients[, "Estimate"]
)
print(df_coef)
tab_sub$yieldPerAcre <- pred

# compare distributions
dists <- rbind(summary(train$yieldPerAcre), summary(val$yieldPerAcre), summary(pred))
rownames(dists) <- c("2022", "2023", "sub")
print(dists)
write.csv(tab_sub, "output/submission.csv",row.names = F)

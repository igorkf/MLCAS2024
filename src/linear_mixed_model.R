library(tidyverse)
library(lme4)

RMSE <- function(y, ypred) {
  sqrt((sum((y - ypred) ^ 2) / length(y)))
}

val <- read.csv("output/val.csv")
val$genotype <- as.factor(val$genotype)
val$id <- 1:nrow(val)
train <- read.csv("output/train.csv")
train$genotype <- as.factor(train$genotype)
train$id <- 1:nrow(train)

cats <- c("experiment", "block", "img_id", "genotype", "id")
for (cat in cats) {
  train[, cat] <- as.factor(train[, cat])
  val[, cat] <- as.factor(val[, cat])  
}

# first model
vis <- c(
  colnames(train)[grep("NDVI_", colnames(train))],
  colnames(train)[grep("NDRE_", colnames(train))]
)
fixed_eff <- c(
  # "experiment", "experiment:block",
  vis
)

ggplot(train, aes(x = NDVI_mean_fixed, y = yieldPerAcre, color = location)) +
  geom_point()
ggplot(train, aes(x = NDVI_mean_fixed, y = sqrt(yieldPerAcre), color = location)) +
  geom_point()

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
train[c("parent1", "parent2")] <- str_split_fixed(train$genotype, " X ", 2)
val[c("parent1", "parent2")] <- str_split_fixed(val$genotype, " X ", 2)
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
combinations <- unlist(lapply(1:length(vis), function(n) {
  combn(vis, n, simplify = FALSE)
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

# overlapping between 2022 and 2023
cat("overlapping of parent1 and parent2 among 2022 and 2023\n")
union_p1 <- unique(c(train$parent1, train$parent1))
inter_p1 <- intersect(train$parent1, val$parent1)
cat(length(inter_p1), "overlapping from total of", length(union_p1), "\n")
union_p2 <- unique(c(train$parent2, train$parent2))
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
test <- read.csv("output/test.csv")
tab_sub <- read.csv("data/validation/2023/GroundTruth/val_HIPS_HYBRIDS_2023_V2.3.csv")
stopifnot(all(test$experiment == tab_sub$experiment))
test[c("parent1", "parent2")] <- str_split_fixed(test$genotype, " X ", 2)
pred <- predict(mod_full, newdata = test, allow.new.levels = F)  # all levels are known

# compare estimates
df_coef <- data.frame(
  m_2022 = summary(mod_best)$coefficients[, "Estimate"],
  m_2022_2023 = summary(mod_full)$coefficients[, "Estimate"]
)
df_coef
tab_sub$yieldPerAcre <- pred

# compare distributions
dists <- rbind(summary(train$yieldPerAcre), summary(val$yieldPerAcre), summary(pred))
rownames(dists) <- c("2022", "2023", "sub")
dists
write.csv(tab_sub, "output/submission.csv",row.names = F)


# gaussian mixture model
# library(flexmix)
# gmm1 <- flexmix(yieldPerAcre ~ NDVI_mean_fixed + NDVI_median_fixed + NDVI_sum_fixed + 
#                   NDRE_mean_fixed + NDRE_median_fixed + NDRE_max_fixed + NDRE_sum_fixed | genotype,
#                 data = train, k = 2)
# parameters(gmm1)
# val$yhat_gmm1 <- predict(gmm1, newdata = val)$Comp.1
# val$yhat_gmm2 <- predict(gmm1, newdata = val)$Comp.2
# RMSE(val$yieldPerAcre, (val$yhat_gmm2))


######################
# models

# evaluate <- function(mod, newdata, fixed_var) {
#   # predict for new set
#   
#   # random effect contribution
#   blup <- summary(mod, coef = T)$coef.random
#   blup <- blup[!is.na(blup[, "std.error"]), ]
#   nongenotyped <- rownames(tail(blup, 23))  # fix some blups
#   # blup[rownames(blup) %in% nongenotyped, "solution"] <- (
#   #   mean(blup[!rownames(blup) %in% nongenotyped, , "solution"])
#   # )
#   rownames(blup) <- gsub(".*\\_", "", rownames(blup))
#   newdata_sub <- droplevels(newdata[newdata$genotype %in% rownames(blup), ])
#   Z <- model.matrix(yieldPerAcre ~ genotype, data = newdata_sub)[, -1]  # remove indicator
#   colnames(Z) <- gsub("genotype", "", colnames(Z))
#   blup_sub <- blup[rownames(blup) %in% colnames(Z), ]
#   u <- blup_sub[, "solution", drop = F]
#   Zu <- as.vector(Z %*% u)
#   
#   # fixed effect contribution
#   X <- cbind(
#     matrix(1, nrow = nrow(newdata_sub)),
#     as.matrix(newdata_sub[, fixed_var])
#   )
#   colnames(X)[1] <- "(Intercept)"
#   b <- summary(mod, coef = T)$coef.fixed[, "solution", drop = F]
#   Xb <- as.vector(X %*% b[colnames(X), ])  # multiply respecting order
#   
#   # prediction
#   newdata_sub$id <- rownames(newdata_sub)
#   newdata_sub$yhat <- (Xb + Zu) ** 2
#   
#   # prediction for unknown levels
#   newdata <- newdata %>% 
#     plyr::join(newdata_sub[, c("id", "yhat")], by = "id") %>% 
#     mutate(yhat = ifelse(is.na(yhat), yhat_lm, yhat))
#   
#   rmse <- RMSE(newdata$yieldPerAcre, newdata$yhat)
#   return(list(newdata = newdata, rmse = rmse))
# }
# 
# # fixed effects + random genotype
# mod1 <- asreml(fixed = form, random = ~ genotype, data = train)
# summary(mod1, coef = T)$coef.random
# # plot(mod1)
# results_mod1 <- evaluate(mod1, val, fixed_eff)
# xeval1 <- results_mod1$newdata
# print(results_mod1$rmse)
# cor(xeval1$yieldPerAcre, xeval1$yhat)
# 
# # GBLUP
# G <- as.matrix(read.table("output/G.txt"))
# colnames(G) <- rownames(G)
# mod2 <- asreml(fixed = form, random = ~ vm(genotype, source = G, singG = "NSD"), data = train)
# plot(mod2)
# results_mod2 <- evaluate(mod2, val, fixed_eff)
# xeval2 <- results_mod2$newdata
# print(results_mod2$rmse)
# cor(xeval2$yieldPerAcre, xeval2$yhat)
# 
# # try WxG, with W in plot-level
# create_W <- function(tab) {
#   X <- as.matrix(tab)
#   rownames(X) <- tab$img_id
#   X <- X[, colnames(X) != "img_id"]
#   class(X) <- "numeric"
#   X <- apply(X, 2, function(x) (x - mean(x)) / sd(x))
#   W <- (X %*% t(X)) / length(cols)
#   return(W)
# }
# x <- rbind(train[, c("img_id", vis)], val[, c("img_id", vis)])
# x <- droplevels(x)
# W <- create_W(x)
# dim(W)
# W[1:5, 1:8]
# 
# # model using W
# # mod3 <- asreml(fixed = form, random = ~ vm(img_id, source = W, singG = "NSD"), data = train)
# # plot(mod3)
# # results_mod3 <- evaluate(mod3, val, fixed_eff)
# # xeval2 <- results_mod2$newdata
# # print(results_mod2$rmse)
# # cor(xeval2$yieldPerAcre, xeval2$yhat)

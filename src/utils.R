RMSE <- function(y, ypred) {
  sqrt((sum((y - ypred) ^ 2) / length(y)))
}

bind_block_diag <- function(X, new_levels) {
  Xp <- as.matrix(Matrix::bdiag(X, diag(length(new_levels))))
  rownames(Xp) <- colnames(Xp) <- c(rownames(X), new_levels)
  return(Xp)
}

arc <- function(X) {
  Xp <- as.matrix(bWGR::EigenARC(X))
  rownames(Xp) <- colnames(Xp) <- rownames(X)
  return(Xp)
}

create_random_eff_lin_comb <- function(mod, data, term, trait, verbose) {
  form <- as.formula(paste0("~ ", term, " - 1"))
  Z <- model.matrix(
    form, data = data,
    contrasts.arg = lapply(data[, term, drop = F], contrasts, contrasts = F)
  )
  if (verbose) cat("dim Z (", term, "):", dim(Z), "\n")
  colnames(Z) <- sub(term, "", colnames(Z))
  u <- as.matrix(mod$U[[paste0("u:", term)]][[trait]][colnames(Z)])
  if (verbose) cat("dim u (", term, "):", dim(u), "\n")
  return(list(Z = Z, u = u))
}

build_prediction <- function(mod, fixed_formula, data, trait, add_p1, add_p2, add_geno, add_env, verbose) {
  X <- model.matrix(fixed_formula, data = data)
  if (verbose) cat("dim X:", dim(X), "\n")
  b <- as.matrix(mod$Beta[, 3])
  if (verbose) cat("dim b:", dim(b), "\n")
  rownames(b) <- mod$Beta[, 2]
  yhat <- X %*% b
  
  if (add_p1 == T) {
    parent1 <- create_random_eff_lin_comb(mod, data, "parent1", trait, verbose)
    Z1 <- parent1$Z
    u1 <- parent1$u
    yhat <- yhat + Z1 %*% u1
  }
  
  if (add_p2 == T) {
    parent2 <- create_random_eff_lin_comb(mod, data, "parent2", trait, verbose)
    Z2 <- parent2$Z
    u2 <- parent2$u
    yhat <- yhat + Z2 %*% u2
  }
  
  if (add_geno == T) {
    genotype <- create_random_eff_lin_comb(mod, data, "genotype", trait, verbose)
    Z3 <- genotype$Z
    u3 <- genotype$u
    yhat <- yhat + Z3 %*% u3
  }
  
  if (add_env == T) {
    env_exp <- create_random_eff_lin_comb(mod, data, "env_exp", trait, verbose)
    Z4 <- env_exp$Z
    u4 <- env_exp$u
    yhat <- yhat + Z4 %*% u4
  }
  
  if (trait == "sqrt_y") {
    yhat <- yhat ^ 2
  } else if (trait == "two_thirds_y") {
    yhat <- yhat ^ (3 / 2)
  } else if (trait == "three_quarters_y") {
    yhat <- yhat ^ (4 / 3)
  } else if (trait == "log_y") {
    yhat <- exp(yhat)
  }
  return(as.vector(yhat))
}

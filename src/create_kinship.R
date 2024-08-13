Mode <- function(x) {
  "https://stackoverflow.com/a/8189441/11122513"
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

threads <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK"))
threads <- ifelse(is.na(threads), 8, threads)

# to convert genotype names
mapping <- read.csv("output/geno_mapping.csv")
mapping <- split(mapping$genotype, mapping$genotype_fix)
names_mapping <- names(mapping)
parents <- read.csv("output/hybrids.csv")

# to fill non-genotyped hybrids
all_parents <- read.csv("output/all_parents.csv")

# convert hapmap to numeric
simplePHENOTYPES::create_phenotypes(
  geno_file = "output/maize.hmp.txt",
  add_QTN_num = 1,
  add_effect = 0.2,
  big_add_QTN_effect = 0.9,
  rep = 10,
  h2 = 0.7,
  model = "A",
  home_dir = getwd(),
  out_geno = "numeric",
)

# read hapmap numeric 
hapmap <- data.table::fread("maize_numeric.txt", nThread = threads, data.table = F)
hapmap <- hapmap[, -c(1:5)] + 1
colnames(hapmap) <- unlist(unname(mapping[colnames(hapmap)]))  # map names
indivs <- colnames(hapmap)
hapmap[1:5, 1:7]

# kinship for individuals
# mn <- data.frame(apply(hapmap, 1, function(x) round(mean(x))))
# md <- data.frame(apply(hapmap, 1, Mode))
# ps <- stringr::str_split_fixed(all_parents$genotype, " X ", 2)
# ps <- unique(c(ps[, 1], ps[, 2]))
# ps[ps == ""] <- " "
# cat("creating non-genotyped individuals:")
# for (p in ps) {
#   if (!p %in% indivs) {
#     cat(p, "\n")
#     hapmap[, p] <- mn
#   }
# }

# write GRM for inbreds
G <- AGHmatrix::Gmatrix(as.matrix(t(hapmap)), maf = 0)
write.table(G, "output/G.txt", row.names = T)

# compute hybrids (there must be a better way to do this!)
for (i in 1:nrow(parents)) {
  p <- unname(unlist(parents[i, ]))
  if (p[1] %in% names_mapping) {
    p[1] <- mapping[p[[1]]]
  }
  if (p[2] %in% names_mapping) {
    p[2] <- mapping[p[[2]]]
  }
  p <- unlist(p)
  if (p[1] %in% indivs & p[2] %in% indivs) {
    cat(i, p, "\n")
    hybrid_temp <- data.frame(round(rowMeans(hapmap[, p])))
    colnames(hybrid_temp) <- paste(p, collapse = " X ")
    if (i == 1) {
      hybrids <- hybrid_temp 
    } else {
      hybrids <- cbind(hybrids, hybrid_temp)
    }
  } else {
    cat(i, p, "not present\n")
  }
}

# add non-genotyped hybrids (because either one of the parents were not genotyped)
# mode, median, or mean?
# k <- 0
# for (i in 1:nrow(all_parents)) {
#   h <- all_parents[i, "genotype"]
#   if (!h %in% colnames(hybrids)) {
#     hybrid_temp <- mn
#     colnames(hybrid_temp) <- h
#     hybrids <- cbind(hybrids, hybrid_temp)
#     k <- k + 1
#     cat(i, h, "included\n")
#   }
# }
# cat("# non-genotyped hybrids:", k, "from", nrow(parents), "\n")

# write GRM for hybrids
hybrids[1:5, 1:7]
G1G2 <- AGHmatrix::Gmatrix(as.matrix(t(hybrids)), maf = 0)
write.table(G1G2, "output/G1G2.txt", row.names = T)

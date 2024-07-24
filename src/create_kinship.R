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
hapmap[1:5, 1:10]
hapmap <- hapmap[, -c(1:5)] + 1
indivs <- colnames(hapmap)

# compute hybrids (there must be a better way to do this!)
for (i in 1:nrow(parents)) {
  p <- unname(unlist(parents[i, ]))
  if (p[1] %in% indivs & p[2] %in% indivs) {
    cat(i, p, "\n")
    hybrid_temp <- data.frame(round(rowMeans(hapmap[, p])))
    if (p[1] %in% names_mapping) {
      p[1] <- mapping[p[[1]]]
    }
    if (p[2] %in% names_mapping) {
      p[2] <- mapping[p[[2]]]
    }
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

# add non-genotyped hybrids (because either of the parents were not genotyped)
# mode, median, or mean?
md <- data.frame(apply(hapmap, 1, Mode))
k <- 0
for (i in 1:nrow(all_parents)) {
  h <- all_parents[i, "genotype"]
  if (!h %in% colnames(hybrids)) {
    hybrid_temp <- md
    colnames(hybrid_temp) <- h
    hybrids <- cbind(hybrids, hybrid_temp)
    k <- k + 1
    cat(i, h, "included\n")
  }
}
cat("# non-genotyped hybrids:", k, "from", nrow(parents), "\n")

# write GRM
G <- AGHmatrix::Gmatrix(as.matrix(t(hybrids)), maf = 0)
write.table(G, "output/G.txt", row.names = T)

# read GRM
# a <- as.matrix(read.table("output/G.txt"))
# colnames(a) <- rownames(a)

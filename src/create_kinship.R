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

# to fill non-genotyped hybrids
all_parents <- read.csv("output/all_parents.csv")

hapmap <- data.table::fread("output/maize_numeric.hmp.txt", nThread = threads, data.table = F)
parents <- read.csv("output/hybrids.csv")

# compute hybrids
hybrids <- data.frame()
for (i in 1:nrow(parents)) {
  p <- unname(unlist(parents[i, ]))
  hybrid_temp <- hapmap[hapmap[, "<Marker>"] %in% p, -1] * 2  # {0, 0.5, 1} -> {0, 1, 2}
  hybrid_temp <- as.data.frame.list(round(colMeans(hybrid_temp)))  # {0, 1, 2}
  if (p[1] %in% names_mapping) {
    p[1] <- mapping[p[[1]]]
  }
  if (p[2] %in% names_mapping) {
    p[2] <- mapping[p[[2]]]
  }
  rownames(hybrid_temp) <- paste(p, collapse = " X ")
  hybrids <- rbind(hybrids, hybrid_temp)
}

# add non-genotyped hybrids
md <- as.data.frame.list(apply(hybrids, 2, Mode))
k <- 0
for (i in 1:nrow(all_parents)) {
  h <- all_parents[i, "genotype"]
  if (!h %in% rownames(hybrids)) {
    hybrid_temp <- md
    rownames(hybrid_temp) <- h
    hybrids <- rbind(hybrids, hybrid_temp)
    k <- k + 1
  }
}
cat("# non-genotyped hybrids:", k, "from", nrow(hybrids), "\n")

# write GRM
G <- AGHmatrix::Gmatrix(as.matrix(hybrids), maf = 0)
write.table(G, "output/G.txt", row.names = T)

# read GRM
# a <- as.matrix(read.table("output/G.txt"))
# colnames(a) <- rownames(a)

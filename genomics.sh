# load modules
# module load samtools/1.10 gatk/4.2.6.1

# filter samples
bcftools view -S output/geno_samples.txt data/chr_1_imputed.vcf.gz > output/chr_1_imputed_filt.vcf.gz

# keep only snps and MAF > 0.01
bcftools view -m2 -M2 -v snps -q 0.01:minor output/chr_1_imputed_filt.vcf.gz > output/chr_1_imputed_filt_m001_snps.vcf.gz

# LD pruning and missing filtering
bcftools +prune -w 100 -l 0.9 -e'F_MISSING>=0.5' output/chr_1_imputed_filt_m001_snps.vcf.gz -Oz -o output/chr_1_imputed_filt_m001_snps_pruned.vcf.gz

# create index and table
# 0/0	the sample is a homozygous reference
# 0/1	the sample is heterozygous (carries both reference and alternate alleles)
# 1/1	the sample is a homozygous alternate
# ./.	No genotype called or missing genotype
gatk IndexFeatureFile -I output/chr_1_imputed_filt_m001_snps_pruned.vcf.gz
gatk VariantsToTable -V output/chr_1_imputed_filt_m001_snps_pruned.vcf.gz -F CHROM -F POS -F REF -F ALT -GF GT -O output/chr_1.tsv
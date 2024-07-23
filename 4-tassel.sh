#!/bin/bash

#SBATCH --job-name=tassel
#SBATCH --output=logs/tassel_%j.out
#SBATCH --partition=cloud72
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

## configs 
module purge

## copy needed files to /scratch
cd /scratch/$SLURM_JOB_ID
mkdir -p data src logs output
scp -r $SLURM_SUBMIT_DIR/tassel-5-standalone .
scp $SLURM_SUBMIT_DIR/output/hybrids.txt output
scp $SLURM_SUBMIT_DIR/output/merged_m005_snps_pruned.vcf.gz output

#####################################################
## run tasks
#####################################################

# sort VCF
./tassel-5-standalone/run_pipeline.pl -Xms10g -Xmx60g -SortGenotypeFilePlugin -inputFile output/merged_m005_snps_pruned.vcf.gz -outputFile output/merged_m005_snps_pruned_sorted.vcf.gz -fileType VCF

# convert VCF to Hapmap
./tassel-5-standalone/run_pipeline.pl -Xms10g -Xmx60g -fork1 -vcf output/merged_m005_snps_pruned_sorted.vcf.gz -export output/maize.hmp.txt -exportType Hapmap

# convert Hapmap to numeric
./tassel-5-standalone/run_pipeline.pl -Xms10g -Xmx60g -h output/maize.hmp.txt -NumericalGenotypePlugin -endPlugin -export output/maize_numeric.hmp.txt -exportType ReferenceProbability

#####################################################

## copy needed output files to /home
scp -r output/maize_numeric.hmp.txt $SLURM_SUBMIT_DIR/output/
# scp -r logs/* $SLURM_SUBMIT_DIR/logs/

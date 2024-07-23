#!/bin/bash

#SBATCH --job-name=genom
#SBATCH --output=logs/genomics_%j.out
#SBATCH --partition=cloud72
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00

## configs 
module purge
module load samtools/1.10 intel/21.2.0 mkl/21.3.0 R/4.3.0 gcc/11.2.1 python/2024.02-1-anaconda
conda activate mlcas2024

## copy needed files to /scratch
cd /scratch/$SLURM_JOB_ID
mkdir -p data src logs output
mkdir -p data/train/2022/DataPublication_final/GroundTruth
mkdir -p data/train/2023/DataPublication_final/GroundTruth
mkdir -p data/validation/2023/GroundTruth
scp -r $SLURM_SUBMIT_DIR/data/TPJ_16123_TableS1.xlsx data/
scp -r $SLURM_SUBMIT_DIR/data/*.vcf.gz* data/
scp -r $SLURM_SUBMIT_DIR/data/train/2022/DataPublication_final/GroundTruth/HYBRID_HIPS_V3.5_ALLPLOTS.csv data/train/2022/DataPublication_final/GroundTruth/
scp -r $SLURM_SUBMIT_DIR/data/train/2023/DataPublication_final/GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv data/train/2023/DataPublication_final/GroundTruth/
scp -r $SLURM_SUBMIT_DIR/data/validation/2023/GroundTruth/val_HIPS_HYBRIDS_2023_V2.3.csv data/validation/2023/GroundTruth/
scp $SLURM_SUBMIT_DIR/src/match_hybrids.R src/
scp $SLURM_SUBMIT_DIR/src/constants.py src/
scp $SLURM_SUBMIT_DIR/src/create_hybrids_list.py src/
scp $SLURM_SUBMIT_DIR/MLCAS2024.Rproj .

#####################################################
## run tasks
#####################################################

## generate genotype mapping and hybrids list
Rscript src/match_hybrids.R
python src/create_hybrids_list.py

## combine chromosomes
bcftools concat data/chr_{1..10}_imputed.vcf.gz --threads 8 -Oz -o output/merged.vcf.gz

## keep only few samples, the snps, and filter MAF > 0.05
bcftools view -S output/geno_samples.txt -m2 -M2 -v snps -q 0.05:minor output/merged.vcf.gz -Oz -o output/merged_m005_snps.vcf.gz

## LD pruning and missing filtering
bcftools +prune -w 100 -l 0.9 -e'F_MISSING>=0.5' output/merged_m005_snps.vcf.gz -Oz -o output/merged_m005_snps_pruned.vcf.gz

#####################################################

## copy needed output files to /home
scp -r output/geno_mapping.csv $SLURM_SUBMIT_DIR/output/
scp -r output/all_parents.csv $SLURM_SUBMIT_DIR/output/
scp -r output/hybrids.csv $SLURM_SUBMIT_DIR/output/
scp -r output/merged_m005_snps_pruned.vcf.gz $SLURM_SUBMIT_DIR/output/
# scp -r logs/* $SLURM_SUBMIT_DIR/logs/

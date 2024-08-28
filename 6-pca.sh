#!/bin/bash

#SBATCH --job-name=pca
#SBATCH --output=logs/pca_%j.out
#SBATCH --partition=cloud72
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:60:00

## configs 
module purge
module load python/2024.02-1-anaconda
conda activate mlcas2024

## copy needed files to /scratch
cd /scratch/$SLURM_JOB_ID
mkdir -p data src logs output
scp $SLURM_SUBMIT_DIR/maize_numeric.txt .
scp $SLURM_SUBMIT_DIR/src/pca_snps.py src/

#####################################################
## run tasks
#####################################################

python src/pca_snps.py

#####################################################

## copy needed output files to /home
scp output/pca_snps.csv $SLURM_SUBMIT_DIR/output/
# scp -r logs/* $SLURM_SUBMIT_DIR/logs/

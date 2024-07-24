#!/bin/bash

#SBATCH --job-name=kinship
#SBATCH --output=logs/kinship_%j.out
#SBATCH --partition=cloud72
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

## configs 
module purge
module load intel/21.2.0 mkl/21.3.0 R/4.3.0 gcc/11.2.1

## copy needed files to /scratch
cd /scratch/$SLURM_JOB_ID
mkdir -p data src logs output
scp $SLURM_SUBMIT_DIR/output/geno_mapping.csv output/
scp $SLURM_SUBMIT_DIR/output/all_parents.csv output/
scp $SLURM_SUBMIT_DIR/output/hybrids.csv output/
scp $SLURM_SUBMIT_DIR/src/create_kinship.R src/
scp $SLURM_SUBMIT_DIR/maize_numeric.txt .
scp $SLURM_SUBMIT_DIR/MLCAS2024.Rproj .

#####################################################
## run tasks
#####################################################

Rscript src/create_kinship.R

#####################################################

## copy needed output files to /home
scp -r output/G.txt $SLURM_SUBMIT_DIR/output/
# scp -r logs/* $SLURM_SUBMIT_DIR/logs/

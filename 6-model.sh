#!/bin/bash

#SBATCH --job-name=model
#SBATCH --output=logs/model_%j.out
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
mkdir -p data/validation/2023/GroundTruth/
scp $SLURM_SUBMIT_DIR/output/train.csv output/
scp $SLURM_SUBMIT_DIR/output/val.csv output/
scp $SLURM_SUBMIT_DIR/output/test.csv output/
scp $SLURM_SUBMIT_DIR/output/G.txt output/
scp $SLURM_SUBMIT_DIR/data/validation/2023/GroundTruth/val_HIPS_HYBRIDS_2023_V2.3.csv data/validation/2023/GroundTruth/
scp $SLURM_SUBMIT_DIR/src/linear_mixed_model.R src/
scp $SLURM_SUBMIT_DIR/src/utils.R src/
scp $SLURM_SUBMIT_DIR/MLCAS2024.Rproj .

#####################################################
## run tasks
#####################################################

Rscript src/linear_mixed_model.R > output/results.txt

#####################################################

## copy needed output files to /home
scp -r output/submission.csv $SLURM_SUBMIT_DIR/output/
scp -r output/results.txt $SLURM_SUBMIT_DIR/output/
# scp -r logs/* $SLURM_SUBMIT_DIR/logs/

#!/bin/bash

#SBATCH --job-name=datasets
#SBATCH --output=logs/datasets_%j.out
#SBATCH --partition=cloud72
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00

## configs 
module purge
module load python/2024.02-1-anaconda
conda activate mlcas2024

## copy needed files to /scratch
cd /scratch/$SLURM_JOB_ID
mkdir -p data src logs output
mkdir -p data/train/2022/DataPublication_final/GroundTruth
mkdir -p data/train/2023/DataPublication_final/GroundTruth
mkdir -p data/validation/2023/GroundTruth
mkdir -p data/test/Test/Test/GroundTruth
scp -r $SLURM_SUBMIT_DIR/data/train/2022/DataPublication_final/GroundTruth/HYBRID_HIPS_V3.5_ALLPLOTS.csv data/train/2022/DataPublication_final/GroundTruth/
scp -r $SLURM_SUBMIT_DIR/data/train/2023/DataPublication_final/GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv data/train/2023/DataPublication_final/GroundTruth/
scp -r $SLURM_SUBMIT_DIR/data/validation/2023/GroundTruth/val_HIPS_HYBRIDS_2023_V2.3.csv data/validation/2023/GroundTruth/
scp -r $SLURM_SUBMIT_DIR/data/test/Test/Test/GroundTruth/test_HIPS_HYBRIDS_2023_V2.3.csv data/test/Test/Test/GroundTruth/test_HIPS_HYBRIDS_2023_V2.3.csv
scp -r $SLURM_SUBMIT_DIR/output/satellite*.csv output/
scp $SLURM_SUBMIT_DIR/src/constants.py src/
scp $SLURM_SUBMIT_DIR/src/create_datasets.py src/

#####################################################
## run tasks
#####################################################

python src/create_datasets.py

#####################################################

## copy needed output files to /home
scp -r output/* $SLURM_SUBMIT_DIR/output/

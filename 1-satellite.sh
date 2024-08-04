#!/bin/bash

#SBATCH --job-name=sat_phe
#SBATCH --output=logs/sat_pheno_%j.out
#SBATCH --partition=cloud72
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

## configs 
module purge
module load python/2024.02-1-anaconda
conda activate mlcas2024

## copy needed files to /scratch
cd /scratch/$SLURM_JOB_ID
mkdir -p data src logs output
mkdir -p data/train/2022/DataPublication_final/Satellite
mkdir -p data/train/2023/DataPublication_final/Satellite
mkdir -p data/validation/2023/Satellite
scp -r $SLURM_SUBMIT_DIR/data/train/2022/DataPublication_final/Satellite data/train/2022/DataPublication_final
scp -r $SLURM_SUBMIT_DIR/data/train/2023/DataPublication_final/Satellite data/train/2023/DataPublication_final
scp -r $SLURM_SUBMIT_DIR/data/validation/2023/Satellite data/validation/2023
scp $SLURM_SUBMIT_DIR/src/process_satellite.py src/

#####################################################
## run tasks
#####################################################

python -u src/process_satellite.py --data=train --year=2022
python -u src/process_satellite.py --data=train --year=2023
python -u src/process_satellite.py --data=validation --year=2023

#####################################################

## copy needed output files to /home
scp -r output/* $SLURM_SUBMIT_DIR/output/
# scp -r logs/* $SLURM_SUBMIT_DIR/logs/

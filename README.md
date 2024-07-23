# MLCAS2024

## Install packages
1. Install Python packages:
    ```
    conda env create -f environment.yml
    conda activate mlcas2024
    ```

2. Install R packages (if needed):
    ```
    install.packages("...")
    ```

## Satellite and phenotypic data
1. Create satellite features:
    ```
    python src/process_satellite.py --data=train --year=2022
    python src/process_satellite.py --data=train --year=2023
    python src/process_satellite.py --data=validation --year=2023
    ```

2. Create folds:
    ```
    python src/create_folds.py
    ```

## Genotipic data
1. Download Supporting Information from [here](https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1111%2Ftpj.16123&file=tpj16123-sup-0001-Supinfo.zip) and put it on the data folder (unzipped).

2. Download the genomic data from [here](https://ars-usda.app.box.com/v/maizegdb-public/folder/189779501832) and put it on the data folder as well (unzipped).

3. Run bioinformatics pipeline (you will need `samtools 1.10` and `R`):
    ```
    ./genomics.sh
    ```
    This might take a while because there are around 46 million SNPs from 1515 samples. After filtering, we kept 46 samples and 5523128 SNPs.

4. Configure TASSEL and create hybrids (version used: 5.2.93):
    ```
    git clone https://bitbucket.org/tasseladmin/tassel-5-standalone.git
    ./tassel-5-standalone/run_pipeline.pl -h mdp_genotype.hmp.txt -CreateHybridGenotypesPlugin -hybridFile hybrids.txt -endPlugin -export output
    ```
    Guide: https://bytebucket.org/tasseladmin/tassel-5-source/wiki/docs/Tassel5PipelineCLI.pdf

5. Create genomic relationship matrix:
    ```
    sbatch 5-kinship.sh
    ```
    
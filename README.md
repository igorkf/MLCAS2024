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
    This might take a while because there are 46 million SNPs.
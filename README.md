# MLCAS2024

Information about the competition:   
https://eval.ai/web/challenges/challenge-page/2332/overview

## Setup data and environment
Clone the repository.

Download the data and put it on the `data` folder:   
https://iastate.box.com/s/p8nj1ukvwx3yo7off8y8yspdruc0mjna

Install Python packages and activate conda environment:
```
conda env create -f environment.yml
conda activate mlcas2024
```

## Preprocess the data
Create satellite features:
```
python -u src/process_satellite.py --data=train --year=2022
python -u src/process_satellite.py --data=train --year=2023
python -u src/process_satellite.py --data=validation --year=2023
python -u src/process_satellite.py --data=test --year=2023
```

Create datasets:
```
python -u src/create_datasets.py
```

<!-- ## Genotipic data
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
    ``` -->

## Fit model
Install R packages:
```
install.packages("tidyverse")
install.packages("lme4")
```

Fit a Linear Mixed Model:
```
Rscript src/blup.R > output/results.txt
```

Predictions will be available at `output/submission.csv`. Log of the results will be at `output/results.txt`.

****

Notes:   
Everything was done on a CentOS 7 cluster, so you can ignore all the shell scripts because they were used to schedule jobs on the cluster.

There are some additional files not being used due to some previous  experimentations. You can ignore them.

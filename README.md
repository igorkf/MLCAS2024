# MLCAS2024

Information about the competition:   
https://eval.ai/web/challenges/challenge-page/2332/overview

## Setup data and environment
Clone the repository.

Download the data (https://iastate.box.com/s/p8nj1ukvwx3yo7off8y8yspdruc0mjna).   

Put the unzipped folders `train`, `validation`, and `test` on the `data` folder.

I realized that depending on your OS, the folders might get unzipped differently.       
Thus, make sure the data structure (after unzipping the folders) is as follows:
```
data/train/2022/...
data/train/2023/...
data/validation/2023/...
data/test/Test/Test/...
```

Create a conda environment and install Python packages:
```
conda create -n mlcas2024 python=3.11
conda activate mlcas2024
conda install pandas rasterio tqdm
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
_Note: if you can't run this code from your terminal, just run it from RStudio itself or from an RStudio terminal._

Predictions will be available at `output/submission.csv`. Log of the results will be at `output/results.txt`.


******


Tested on:   
- Windows 10
- CentOS 7

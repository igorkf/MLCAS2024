# MLCAS2024

Information about the competition:   
https://eval.ai/web/challenges/challenge-page/2332/overview

## Setup data and environment
Clone the repository.

Download the data and put it on the `data` folder (unzip folders as needed):   
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

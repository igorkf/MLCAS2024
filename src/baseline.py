from pathlib import Path

import pandas as pd
import numpy as np


if __name__ == '__main__':
    PATH_TRAIN = Path('data/train/2022/DataPublication_final/GroundTruth')
    PATH_TRAIN_2023 = Path('data/train/2023/DataPublication_final/GroundTruth')
    PATH_VAL = Path('data/validation/2023/GroundTruth')
    df_val = pd.read_csv(PATH_VAL / 'val_HIPS_HYBRIDS_2023_V2.3.csv').drop('yieldPerAcre', axis=1)
    df_train = pd.read_csv(PATH_TRAIN / 'HYBRID_HIPS_V3.5_ALLPLOTS.csv')
    # df_train = df_train[df_train['location'].isin(df_val['location'])].reset_index(drop=True)
    df_train_2023 = pd.read_csv(PATH_TRAIN_2023 / 'train_HIPS_HYBRIDS_2023_V2.3.csv').rename(columns={'yieldPerAcre': 'ytrue'})

    geno_train = set(df_train['genotype'])
    print('# unique genotypes in train:', len(geno_train))
    geno_val = set(df_val['genotype'])
    print('# unique genotypes in val:', len(geno_val))
    print('# unique genoypes in val but not in train:', len(geno_val - geno_train))

    grouping_cols = ['location', 'irrigationProvided', 'nitrogenTreatment', 'poundsOfNitrogenPerAcre', 'genotype']
    median_per_group = (
        df_train
        .groupby(grouping_cols)['yieldPerAcre']
        .median()
        .rename('level_0')
    )
    df_train_2023 = df_train_2023.merge(median_per_group, on=grouping_cols, how='left')
    df_val = df_val.merge(median_per_group, on=grouping_cols, how='left')
    print('# NA:', df_val['level_0'].isna().sum(), '/', len(df_val))
    
    grouping_cols = ['location', 'irrigationProvided', 'nitrogenTreatment', 'poundsOfNitrogenPerAcre']
    median_per_group = (
        df_train
        .groupby(grouping_cols)['yieldPerAcre']
        .median()
        .rename('level_1')
    )
    df_train_2023 = df_train_2023.merge(median_per_group, on=grouping_cols, how='left')
    df_val = df_val.merge(median_per_group, on=grouping_cols, how='left')
    print('# NA:', df_val['level_1'].isna().sum(), '/', len(df_val))

    grouping_cols = ['location']
    median_per_group = (
        df_train
        .groupby(grouping_cols)['yieldPerAcre']
        .median()
        .rename('level_2')
    )
    df_train_2023 = df_train_2023.merge(median_per_group, on=grouping_cols, how='left')
    df_val = df_val.merge(median_per_group, on=grouping_cols, how='left')
    print('# NA:', df_val['level_2'].isna().sum(), '/', len(df_val))

    # validate on training (2023)
    df_train_2023['yieldPerAcre'] = pd.NA
    df_train_2023['yieldPerAcre'] = np.where(df_train_2023['level_0'].notna(), df_train_2023['level_0'], df_train_2023['level_1'])
    df_train_2023['yieldPerAcre'] = df_train_2023['yieldPerAcre'].fillna(df_train_2023['level_2']).round(2)
    rmse = ((df_train_2023['ytrue'] - df_train_2023['yieldPerAcre']) ** 2).mean() ** 0.5
    print('RMSE:', round(rmse, 3), '\n')

    # prediction
    df_val['yieldPerAcre'] = pd.NA
    df_val['yieldPerAcre'] = np.where(df_val['level_0'].notna(), df_val['level_0'], df_val['level_1'])
    df_val['yieldPerAcre'] = df_val['yieldPerAcre'].fillna(df_val['level_2']).round(2)
    df_val = df_val.drop(['level_0', 'level_1', 'level_2'], axis=1)
    print('# NA:', df_val['yieldPerAcre'].isna().sum(), '/', len(df_val))
    assert df_val['yieldPerAcre'].isna().sum() == 0
    df_val.to_csv('output/median_by_groups.csv', index=False)
    
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


PATH_TRAIN_2022 = Path("data/train/2022/DataPublication_final/GroundTruth")
PATH_TRAIN_2023 = Path("data/train/2023/DataPublication_final/GroundTruth")
PATH_VAL = Path("data/validation/2023/GroundTruth")
DESIGN_COLS = ["location", "experiment", "range", "row"]

if __name__ == "__main__":

    # read field data
    df_train_2022 = pd.read_csv(
        PATH_TRAIN_2022 / "HYBRID_HIPS_V3.5_ALLPLOTS.csv"
    ).dropna(subset=["yieldPerAcre"])
    df_train_2023 = pd.read_csv(PATH_TRAIN_2023 / "train_HIPS_HYBRIDS_2023_V2.3.csv")
    df_train = pd.concat([df_train_2022, df_train_2023], ignore_index=True)
    df_test = pd.read_csv(PATH_VAL / "val_HIPS_HYBRIDS_2023_V2.3.csv").drop(
        "yieldPerAcre", axis=1
    )
    df_sub = df_test.copy()

    # read BLUPS and merge
    blups = pd.read_csv("output/blups.csv").iloc[:, :3]
    blups.columns = ["genotype", "blup", "blup_stderror"]
    blups["genotype"] = blups["genotype"].str.replace("genotype_", "")
    df_train = df_train.merge(blups, on="genotype", how="left")
    df_test = df_test.merge(blups, on="genotype", how="left")
    BLUP_COLS = [
        "blup", 
        # "blup_stderror"
    ]

    # read satellite data and merge
    df_train_sat_2022 = pd.read_csv("output/satellite_train_2022.csv")
    df_train_sat_2023 = pd.read_csv("output/satellite_train_2023.csv")
    df_train_sat = pd.concat([df_train_sat_2022, df_train_sat_2023], ignore_index=True)
    df_train = df_train.merge(df_train_sat, on=DESIGN_COLS, how="left")
    df_test_sat = pd.read_csv("output/satellite_validation_2023.csv")
    df_test = df_test.merge(df_test_sat, on=DESIGN_COLS, how="left")
    SAT_COLS = df_train_sat.filter(regex="TP1|TP2|TP3", axis=1).columns.tolist()

    # categorical columns
    CAT_COLS = [
        "location",
        "nitrogenTreatment"
    ]
    ohe = OneHotEncoder(sparse_output=False)
    df_train_cat = pd.DataFrame(
        ohe.fit_transform(df_train[CAT_COLS]), columns=ohe.get_feature_names_out()
    )
    df_train = pd.concat([df_train_cat, df_train], axis=1)
    df_test_cat = pd.DataFrame(
        ohe.transform(df_test[CAT_COLS]), columns=ohe.get_feature_names_out()
    )
    df_test = pd.concat([df_test_cat, df_test], axis=1)
    CAT_COLS = df_train_cat.columns.tolist()

    FEATURES = [
        # *CAT_COLS,
        # *SAT_COLS,
        *BLUP_COLS,
    ]

    # split
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = np.zeros((kf.n_splits,))
    for fold, (tr, val) in enumerate(
        kf.split(df_train.drop("yieldPerAcre", axis=1), df_train["yieldPerAcre"])
    ):
        print("Fold:", fold)
        xtrain = df_train.loc[tr, FEATURES]
        ytrain = df_train.loc[tr, "yieldPerAcre"]
        xval = df_train.loc[val, FEATURES]
        yval = df_train.loc[val, "yieldPerAcre"]
        xtest = df_test.loc[:, FEATURES]

        # merge climate data
        # xtrain_clim = pd.read_csv("output/climate_train_2022.csv")
        # clim_cols = xtrain_clim.drop("location", axis=1).columns.tolist()
        # xtrain = df_train.merge(xtrain_clim, on="location")[clim_cols]
        # xval_clim = pd.read_csv("output/climate_train_2023.csv")
        # xval = df_val.merge(xval_clim, on="location")[clim_cols]
        # xtest_clim = pd.read_csv("output/climate_validation_2023.csv")
        # xtest = df_test.merge(xtest_clim, on="location")[clim_cols]

        # fit
        model = RandomForestRegressor(random_state=42, max_depth=3)
        model.fit(xtrain, ytrain)

        # evaluate
        p = model.predict(xval)
        rmse = ((yval - p) ** 2).mean() ** 0.5
        rmses[fold] = rmse
        print("RMSE:", round(rmse, 3), "\n")

        # predict
        df_sub[f"fold{fold}"] = model.predict(xtest)

    # score
    print("-" * 20)
    print("Mean:", np.mean(rmses).round(3))
    print("Std:", np.std(rmses).round(3))
    print("-" * 20)

    # submission
    FOLD_COLS = df_sub.filter(regex="fold[0-9]", axis=1).columns.tolist()
    df_sub["yieldPerAcre"] = df_sub[FOLD_COLS].mean(axis=1)
    df_sub = df_sub.drop(FOLD_COLS, axis=1)
    print(
        pd.concat(
            [
                df_train[["yieldPerAcre"]]
                .describe()
                .rename(columns={"yieldPerAcre": "obs"})
                .T,
                df_sub[["yieldPerAcre"]]
                .describe()
                .rename(columns={"yieldPerAcre": "pred"})
                .T,
            ]
        )
    )
    df_sub.to_csv("output/submission.csv", index=False)

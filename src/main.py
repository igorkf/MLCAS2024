from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder


def pivot(df):
    df = df.pivot(index=["location", "experiment", "range", "row"], columns=["tp"])
    df.columns = [f"{x[0]}_{x[1]}" for x in df.columns]
    df = df.reset_index()
    return df


def fix_timepoint(df, location, old_col, new_col):
    df.loc[df["location"] == location, new_col] = df.loc[
        df["location"] == location, old_col
    ]


PATH_TRAIN_2022 = Path("data/train/2022/DataPublication_final/GroundTruth")
PATH_TRAIN_2023 = Path("data/train/2023/DataPublication_final/GroundTruth")
PATH_VAL = Path("data/validation/2023/GroundTruth")
DESIGN_COLS = ["location", "experiment", "range", "row"]

if __name__ == "__main__":

    # read field data
    df_train_2022 = pd.read_csv(
        PATH_TRAIN_2022 / "HYBRID_HIPS_V3.5_ALLPLOTS.csv"
    ).dropna(subset=["yieldPerAcre"])
    df_2023 = pd.read_csv(PATH_TRAIN_2023 / "train_HIPS_HYBRIDS_2023_V2.3.csv")
    df_test = pd.read_csv(PATH_VAL / "val_HIPS_HYBRIDS_2023_V2.3.csv").drop(
        "yieldPerAcre", axis=1
    )
    df_sub = df_test.copy()

    # read BLUPS and merge
    # blups = pd.read_csv("output/blups.csv").iloc[:, :3]
    # blups.columns = ["genotype", "blup", "blup_stderror"]
    # blups["genotype"] = blups["genotype"].str.replace("genotype_", "")
    # df_train = df_train_2022.merge(blups, on="genotype", how="left")
    # df_test = df_test.merge(blups, on="genotype", how="left")
    # BLUP_COLS = [
    #     "blup",
    #     # "blup_stderror"
    # ]

    # read satellite data and merge
    df_train_sat_2022 = pivot(pd.read_csv("output/satellite_train_2022.csv"))
    df_train_2022 = df_train_2022.merge(df_train_sat_2022, on=DESIGN_COLS, how="left")
    df_sat_2023 = pivot(pd.read_csv("output/satellite_train_2023.csv"))
    df_2023 = df_2023.merge(df_sat_2023, on=DESIGN_COLS, how="left")
    df_test_sat = pivot(pd.read_csv("output/satellite_validation_2023.csv"))
    df_test = df_test.merge(df_test_sat, on=DESIGN_COLS, how="left")

    # fix timepoints for each location
    # commented is difference of days between planting date and chosen TP
    FUNCS = ["median"]
    for func in FUNCS:
        new_col = f"NDVI_{func}_fixed"

        # train2022
        fix_timepoint(df_train_2022, "Scottsbluff", f"NDVI_{func}_TP3", new_col)  # 79
        fix_timepoint(df_train_2022, "Lincoln", f"NDVI_{func}_TP2", new_col)  # 75
        fix_timepoint(df_train_2022, "MOValley", f"NDVI_{func}_TP2", new_col)  # 82
        fix_timepoint(df_train_2022, "Ames", f"NDVI_{func}_TP3", new_col)  # 79
        fix_timepoint(df_train_2022, "Crawfordsville", f"NDVI_{func}_TP3", new_col)  # 82

        # train2023
        fix_timepoint(df_2023, "Lincoln", f"NDVI_{func}_TP3", new_col)  # 88
        fix_timepoint(df_2023, "MOValley", f"NDVI_{func}_TP1", new_col)  # 79

        # val2023
        fix_timepoint(df_test, "Ames", f"NDVI_{func}_TP2", new_col)  # 72

    SAT_COLS = df_train_2022.filter(regex="_fixed", axis=1).columns.tolist()

    # categorical columns
    CAT_COLS = [
        # "location",
        "nitrogenTreatment"
    ]
    ohe = OneHotEncoder(sparse_output=False)
    df_train_2022_cat = pd.DataFrame(
        ohe.fit_transform(df_train_2022[CAT_COLS]), columns=ohe.get_feature_names_out()
    )
    df_train_2022 = pd.concat([df_train_2022_cat, df_train_2022], axis=1)
    df_2023_cat = pd.DataFrame(
        ohe.transform(df_2023[CAT_COLS]), columns=ohe.get_feature_names_out()
    )
    df_2023 = pd.concat([df_2023_cat, df_2023], axis=1)
    df_test_cat = pd.DataFrame(
        ohe.transform(df_test[CAT_COLS]), columns=ohe.get_feature_names_out()
    )
    df_test = pd.concat([df_test_cat, df_test], axis=1)
    CAT_COLS = df_train_2022_cat.columns.tolist()

    FEATURES = [
        # *CAT_COLS,
        *SAT_COLS,
        # *BLUP_COLS,
    ]

    # split
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = np.zeros((kf.n_splits,))
    for fold, (tr, _) in enumerate(
        kf.split(
            df_train_2022.drop("yieldPerAcre", axis=1), df_train_2022["yieldPerAcre"]
        )
    ):
        print("Fold:", fold)
        xtrain = df_train_2022.loc[tr, FEATURES]
        ytrain = df_train_2022.loc[tr, "yieldPerAcre"]
        xval = df_2023.loc[:, FEATURES]
        yval = df_2023.loc[:, "yieldPerAcre"]
        xtest = df_test.loc[:, FEATURES]

        # checking correlations
        xt = pd.concat([xtrain, ytrain], axis=1)
        corrt = xt.corr()["yieldPerAcre"].rename("train")
        xv = pd.concat([xval, yval], axis=1)
        corrv = xv.corr()["yieldPerAcre"].rename("val")
        corr = pd.concat([corrt, corrv], axis=1).sort_values("train")
        corr['abs_diff'] = (corr['train'] - corr['val']).abs()
        print(corr)

        # merge climate data
        # xtrain_clim = pd.read_csv("output/climate_train_2022.csv")
        # clim_cols = xtrain_clim.drop("location", axis=1).columns.tolist()
        # xtrain = df_train.merge(xtrain_clim, on="location")[clim_cols]
        # xval_clim = pd.read_csv("output/climate_train_2023.csv")
        # xval = df_val.merge(xval_clim, on="location")[clim_cols]
        # xtest_clim = pd.read_csv("output/climate_validation_2023.csv")
        # xtest = df_test.merge(xtest_clim, on="location")[clim_cols]

        # impute
        for col in FEATURES:
            filler = xtrain[col].median()
            xtrain[col] = xtrain[col].fillna(filler)
            xval[col] = xval[col].fillna(filler)
            xtest[col] = xtest[col].fillna(filler)

        # fit
        # model = RandomForestRegressor(random_state=42)
        model = linear_model.LinearRegression()
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
                df_train_2022[["yieldPerAcre"]]
                .describe()
                .rename(columns={"yieldPerAcre": "obs"})
                .T,
                df_2023[["yieldPerAcre"]]
                .describe()
                .rename(columns={"yieldPerAcre": "pred"})
                .T,
                df_sub[["yieldPerAcre"]]
                .describe()
                .rename(columns={"yieldPerAcre": "sub"})
                .T,
            ]
        )
    )
    df_sub.to_csv("output/submission.csv", index=False)

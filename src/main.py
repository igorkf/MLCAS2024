from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt

from constants import *


pd.set_option("display.max_rows", 1000)


def pivot(df):
    df = df.drop(["path", "id"], axis=1).pivot(
        index=["location", "experiment", "range", "row"], columns=["tp"]
    )
    df.columns = [f"{x[0]}_{x[1]}" for x in df.columns]
    df = df.reset_index()
    return df


def fix_timepoint(df, location, old_col, new_col):
    df.loc[df["location"] == location, new_col] = df.loc[
        df["location"] == location, old_col
    ]


DESIGN_COLS = ["location", "experiment", "range", "row"]

if __name__ == "__main__":

    # read field data
    df_train_2022 = pd.read_csv(
        PATH_TRAIN_2022 / "HYBRID_HIPS_V3.5_ALLPLOTS.csv"
    ).dropna(subset=["yieldPerAcre"])
    df_train_2022["experiment"] = df_train_2022["experiment"].str.replace(
        "Hyrbrids", "Hybrids"
    )  # fix typo in Scottsbluff
    df_2023 = pd.read_csv(PATH_TRAIN_2023 / "train_HIPS_HYBRIDS_2023_V2.3.csv")
    df_test = pd.read_csv(PATH_VAL / "val_HIPS_HYBRIDS_2023_V2.3.csv").drop(
        "yieldPerAcre", axis=1
    )
    df_sub = df_test.copy()

    # read G2F individuals
    # indiv = pd.read_csv("data/individuals.csv", header=None)
    # indiv.columns = ['genotype']
    # indiv['genotype'] = indiv['genotype'].str.replace('/', ' X ')
    # indiv['in_G2F'] = 1
    # df_train_2022 = df_train_2022.merge(indiv, on='genotype', how='left')

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

    # merge satellite data
    df_train_sat_2022 = pivot(pd.read_csv("output/satellite_train_2022.csv"))
    df_train_2022 = df_train_2022.merge(df_train_sat_2022, on=DESIGN_COLS, how="left")
    df_sat_2023 = pivot(pd.read_csv("output/satellite_train_2023.csv"))
    df_2023 = df_2023.merge(df_sat_2023, on=DESIGN_COLS, how="left")
    df_test_sat = pivot(pd.read_csv("output/satellite_validation_2023.csv"))
    df_test = df_test.merge(df_test_sat, on=DESIGN_COLS, how="left")

    # merge climate data
    # df_train_2022_clim = pd.read_csv("output/climate_train_2022.csv")
    # CLIM_COLS = df_train_2022_clim.drop("location", axis=1).columns.tolist()
    # df_train_2022 = df_train_2022.merge(df_train_2022_clim, on="location", how='left')
    # df_2023_clim = pd.read_csv("output/climate_train_2023.csv")
    # df_2023 = df_2023.merge(df_2023_clim, on="location", how='left')
    # df_test_clim = pd.read_csv("output/climate_validation_2023.csv")
    # df_test = df_test.merge(df_test_clim, on="location", how='left')
    # CLIM_COLS = [x for x in CLIM_COLS if x in ["T2M_sd_w12-w13"]]

    # fix timepoints for each location
    # the comment is the difference of days between planting date and the chosen TP (i.e. 1, 2, or 3)
    VIS = [
        "_".join(x.split("_")[:2]) for x in df_train_sat_2022 if x not in DESIGN_COLS
    ]
    VIS = [x for x in VIS if "NDVI_max" not in x]  # I guess it has outliers?
    VIS = [x for x in VIS if "EVI" not in x]
    # VIS = [x for x in VIS if "EVI_max" not in x and "EVI_min" not in x and "EVI_median" not in x]
    for vi in VIS:
        new_col = f"{vi}_fixed"

        # train2022
        fix_timepoint(df_train_2022, "Scottsbluff", f"{vi}_TP3", new_col)  # 79
        fix_timepoint(df_train_2022, "Lincoln", f"{vi}_TP2", new_col)  # 75
        fix_timepoint(df_train_2022, "MOValley", f"{vi}_TP2", new_col)  # 82
        fix_timepoint(df_train_2022, "Ames", f"{vi}_TP3", new_col)  # 79
        fix_timepoint(df_train_2022, "Crawfordsville", f"{vi}_TP3", new_col)  # 82

        # train2023
        fix_timepoint(df_2023, "Lincoln", f"{vi}_TP3", new_col)  # 88
        fix_timepoint(df_2023, "MOValley", f"{vi}_TP1", new_col)  # 79

        # val2023
        fix_timepoint(df_test, "Ames", f"{vi}_TP2", new_col)  # 72

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

    # remove Scottsbluff (not representative)
    # df_train_2022 = df_train_2022[
    #     df_train_2022["location"] != "Scottsbluff"
    # ].reset_index(drop=True)

    FEATURES = [
        # *CAT_COLS,
        *SAT_COLS,
        # *CLIM_COLS,
        # *BLUP_COLS,
    ]

    # split
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = np.zeros((kf.n_splits,))
    rs = np.zeros((kf.n_splits,))
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
        corr["abs_diff"] = (corr["train"] - corr["val"]).abs()
        print(corr)

        # remove inconsistent features
        # corr = corr[(corr["abs_diff"] < 0.2) & (corr["train"] > 0.55)]
        # keep_cols = [x for x in corr.index if x != "yieldPerAcre"]
        # xtrain = xtrain[keep_cols]
        # xval = xval[keep_cols]
        # xtest = xtest[keep_cols]
        # print(corr)

        # impute
        # for col in xtrain.columns:
        #     filler = xtrain[col].median()
        #     xtrain[col] = xtrain[col].fillna(filler)
        #     xval[col] = xval[col].fillna(filler)
        #     xtest[col] = xtest[col].fillna(filler)

        # remove outliers using covariance
        # env = EllipticEnvelope(random_state=42)
        # env.fit(xtrain)
        # results = env.predict(xtrain)
        # outliers = results == -1
        # print(pd.Series(results).value_counts())
        # xtrain = xtrain.loc[~outliers].reset_index(drop=True)
        # ytrain = ytrain.loc[~outliers].reset_index(drop=True)
        # print('Dropping', sum(outliers), 'outliers.')

        # fit
        # model = RandomForestRegressor(random_state=42)
        # model = linear_model.Ridge(random_state=42, alpha=0.01)
        model = linear_model.LinearRegression()
        # model = svm.SVR(kernel='linear')
        # model = linear_model.Lasso(random_state=42)
        model.fit(xtrain, ytrain)

        # evaluate
        p = model.predict(xval)
        rmse = ((yval - p) ** 2).mean() ** 0.5
        rmses[fold] = rmse
        r = np.corrcoef(yval, p)[0, 1].round(3)
        rs[fold] = r
        print("RMSE:", round(rmse, 3))
        print("r:", r, "\n")

        # predict
        df_sub[f"fold{fold}"] = model.predict(xtest)

    # score
    print("-" * 20)
    print("RMSE (mean):", np.mean(rmses).round(3))
    print("RMSE (std):", np.std(rmses).round(3))
    print("r (mean)", np.mean(rs).round(3))
    print("r (std)", np.std(rs).round(3))
    print("-" * 20)

    # store everything for git documentation
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 150)

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
    # plt.scatter(yval, p)
    # plt.show()
    df_sub.to_csv("output/submission.csv", index=False)

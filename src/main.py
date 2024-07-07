from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


if __name__ == "__main__":
    PATH_TRAIN_2022 = Path("data/train/2022/DataPublication_final/GroundTruth")
    PATH_TRAIN_2023 = Path("data/train/2023/DataPublication_final/GroundTruth")
    PATH_VAL = Path("data/validation/2023/GroundTruth")
    df_train = pd.read_csv(PATH_TRAIN_2022 / "HYBRID_HIPS_V3.5_ALLPLOTS.csv").dropna(
        subset=["yieldPerAcre"]
    )
    df_val = pd.read_csv(PATH_TRAIN_2023 / "train_HIPS_HYBRIDS_2023_V2.3.csv")
    df_test = pd.read_csv(PATH_VAL / "val_HIPS_HYBRIDS_2023_V2.3.csv").drop(
        "yieldPerAcre", axis=1
    )
    ytrain = df_train["yieldPerAcre"]
    yval = df_val["yieldPerAcre"]

    # merge climate data
    xtrain_clim = pd.read_csv("output/climate_train_2022.csv")
    clim_cols = xtrain_clim.drop("location", axis=1).columns.tolist()
    xtrain = df_train.merge(xtrain_clim, on="location")[clim_cols]
    xval_clim = pd.read_csv("output/climate_train_2023.csv")
    xval = df_val.merge(xval_clim, on="location")[clim_cols]
    xtest_clim = pd.read_csv("output/climate_validation_2023.csv")
    xtest = df_test.merge(xtest_clim, on="location")[clim_cols]

    # add location
    CAT_COLS = ["location", "nitrogenTreatment"]
    ohe = OneHotEncoder(sparse_output=False)
    xtrain_cat = pd.DataFrame(
        ohe.fit_transform(df_train[CAT_COLS]), columns=ohe.get_feature_names_out()
    )
    xtrain = pd.concat([xtrain_cat, xtrain], axis=1)
    xval_cat = pd.DataFrame(
        ohe.transform(df_val[CAT_COLS]), columns=ohe.get_feature_names_out()
    )
    xval = pd.concat([xval_cat, xval], axis=1)
    xtest_cat = pd.DataFrame(
        ohe.transform(df_test[CAT_COLS]), columns=ohe.get_feature_names_out()
    )
    xtest = pd.concat([xtest_cat, xtest], axis=1)

    # fit
    model = RandomForestRegressor(random_state=42)
    model.fit(xtrain, ytrain)

    # evaluate
    p = model.predict(xval)
    rmse = ((yval - p) ** 2).mean() ** 0.5
    print("RMSE:", round(rmse, 3), "\n")

    # predict
    df_test["yieldPerAcre"] = model.predict(xtest)
    df_test.to_csv("output/submission.csv", index=False)

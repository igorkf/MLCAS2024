import pandas as pd
import numpy as np
from sklearn import linear_model

from constants import *

pd.set_option("display.max_columns", 100)

DEL_COLS = ["NDVI_min_fixed", "NDRE_min_fixed", "NDRE_max_fixed"]
CAT_COLS = DESIGN_COLS + ["block", "nitrogenTreatment", "img_id", "genotype"]

if __name__ == "__main__":

    train = pd.read_csv(f"output/train.csv")
    val = pd.read_csv("output/val.csv")
    test = pd.read_csv("output/test.csv")

    FEATURES = [x for x in val.columns if "NDVI" in x or "NDRE" and x != "yieldPerAcre"]
    FEATURES = [x for x in FEATURES if x not in CAT_COLS]
    # FEATURES = [x for x in FEATURES if x not in DEL_COLS]

    # checking correlations
    print("Pearson's correlation between features and yield")
    corrt = train[FEATURES + ["yieldPerAcre"]].corr()["yieldPerAcre"].rename("train")
    corrv = val[FEATURES + ["yieldPerAcre"]].corr()["yieldPerAcre"].rename("val")
    corr = pd.concat([corrt, corrv], axis=1).sort_values("train").iloc[:-1,]
    corr["abs_diff"] = (corr["train"] - corr["val"]).abs()
    print(corr, "\n")

    # separate features and target
    xtrain = train[FEATURES]
    print("Model 1 fitting: 2022")
    print("locations (train 2022):", train["location"].unique())
    print("locations (  val 2023):", val["location"].unique())
    print("locations (  sub 2023):", test["location"].unique())
    print("features:", FEATURES)
    print("# features:", len(FEATURES))
    ytrain = train["yieldPerAcre"]
    xval = val[FEATURES]
    yval = val["yieldPerAcre"]
    xtest = test[FEATURES]
    df_sub = test.copy()

    # transformation
    # ytrain = ytrain ** 0.5

    # del_idxs = xtrain[xtrain['location'] != "Scottsbluff"].index
    # xtrain = xtrain.iloc[~del_idxs, ]
    # ytrain = ytrain.iloc[~del_idxs, ]

    # fit
    # model1 = linear_model.Ridge(random_state=42, alpha=0.01)
    model1 = linear_model.LinearRegression()
    model1.fit(xtrain, ytrain)

    # evaluate
    p = model1.predict(xval)
    # p = p ** 2  # back transform
    rmse = ((yval - p) ** 2).mean() ** 0.5
    r = np.corrcoef(yval, p)[0, 1].round(3)
    print("RMSE:", round(rmse, 3))
    print("r:", r, "\n")

    # fit with 2022 + 2023
    print("Model 2 fitting: 2022 + 2023")
    xtrain = pd.concat([xtrain, xval], ignore_index=True)
    ytrain = pd.concat([ytrain, yval], ignore_index=True)
    model2 = linear_model.LinearRegression()
    model2.fit(xtrain, ytrain)
    df_coef = pd.DataFrame()
    df_coef["feature"] = ["Intercept"] + model1.feature_names_in_.tolist()
    df_coef["model1"] = [model1.intercept_] + model1.coef_.tolist()
    df_coef["model2"] = [model2.intercept_] + model2.coef_.tolist()
    df_coef["abs_diff"] = (df_coef["model1"] - df_coef["model2"]).abs()
    print(df_coef, "\n")

    # submission
    df_sub["yieldPerAcre"] = model2.predict(xtest)
    print(
        pd.concat(
            [
                ytrain.describe().to_frame("2022").T,
                yval.describe().to_frame("2023").T,
                df_sub["yieldPerAcre"].describe().to_frame("sub").T,
            ]
        )
    )
    df_sub.to_csv("output/submission.csv", index=False)

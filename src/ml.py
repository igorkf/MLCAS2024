import pandas as pd
import numpy as np
from catboost import Pool, CatBoostRegressor

pd.set_option("display.max_columns", 100)


def preprocess(df):
    # df[["parent1", "parent2"]] = df["genotype"].str.split(" X ", expand=True).fillna("")
    df = df.drop(DROP_COLS, axis=1)
    features = [
        x
        for x in df.columns
        # if "CI" not in x
        # and "MTCI" not in x
        if "yieldPerAcre" not in x
    ]
    if "yieldPerAcre" not in df.columns:
        df["yieldPerAcre"] = 0.0
    return df[features], df["yieldPerAcre"]


DROP_COLS = [
    "location",
    "experiment",
    "range",
    "row",
    "year",
    "irrigationProvided",
    "nitrogenTreatment",
    "poundsOfNitrogenPerAcre",
    "plotLength",
    "img_id",
    # "genotype",
]
CAT_FEATURES = ["genotype"]


if __name__ == "__main__":
    train = pd.read_csv("output/train.csv")
    xtrain, ytrain = preprocess(train)
    train_pool = Pool(xtrain, ytrain, cat_features=CAT_FEATURES)
    val = pd.read_csv("output/val.csv")
    xval, yval = preprocess(val)
    val_pool = Pool(xval, yval, cat_features=CAT_FEATURES)
    test = pd.read_csv("output/test.csv")
    xtest, _ = preprocess(test)
    test_pool = Pool(xtest, cat_features=CAT_FEATURES)
    print("")

    model = CatBoostRegressor(
        iterations=20000,
        random_state=42,
        l2_leaf_reg=0.01,
        learning_rate=0.01,
        depth=4,
        loss_function="RMSE",
    )
    model.fit(train_pool, eval_set=val_pool)
    pred = model.predict(val_pool)
    rmse = np.mean((yval - pred) ** 2) ** 0.5
    r = np.corrcoef(yval, pred)[0, 1]
    print("RMSE:", rmse)
    print("r:", r)

    # sub
    df_sub = pd.read_csv(
        "data/test/Test/Test/GroundTruth/test_HIPS_HYBRIDS_2023_V2.3.csv"
    )
    df_sub["yieldPerAcre"] = model.predict(test_pool)

    print(
        pd.concat(
            [
                ytrain.describe().to_frame("2022").T,
                yval.describe().to_frame("2023").T,
                df_sub["yieldPerAcre"].describe().to_frame("sub").T,
            ]
        )
    )
    df_sub.to_csv("output/submission.csv")

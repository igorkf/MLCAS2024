import pandas as pd

from constants import *


pd.set_option("display.max_rows", 1000)


def pivot(df):
    df = df.drop(["path", "file"], axis=1).pivot(
        index=["location", "experiment", "range", "row"], columns=["tp"]
    )
    df.columns = [f"{x[0]}_{x[1]}" for x in df.columns]
    df = df.reset_index()
    return df


def fix_timepoint(df, location, old_col, new_col):
    df.loc[df["location"] == location, new_col] = df.loc[
        df["location"] == location, old_col
    ]


def create_img_id(df, cols):
    df["img_id"] = df[cols].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
    return df


if __name__ == "__main__":

    # read field data
    df_train_2022 = pd.read_csv(
        PATH_TRAIN_2022 / "HYBRID_HIPS_V3.5_ALLPLOTS.csv"
    ).dropna(subset=["yieldPerAcre"])
    df_train_2022 = create_img_id(df_train_2022, DESIGN_COLS)
    df_train_2022["year"] = 2022
    df_train_2022["experiment"] = df_train_2022["experiment"].str.replace(
        "Hyrbrids", "Hybrids"
    )  # fix typo in Scottsbluff
    df_2023 = pd.read_csv(PATH_TRAIN_2023 / "train_HIPS_HYBRIDS_2023_V2.3.csv")
    df_2023["year"] = 2023
    df_2023 = create_img_id(df_2023, DESIGN_COLS)
    df_test = pd.read_csv(PATH_VAL / "val_HIPS_HYBRIDS_2023_V2.3.csv").drop(
        "yieldPerAcre", axis=1
    )
    df_test = create_img_id(df_test, DESIGN_COLS)
    df_test["year"] = 2023

    # merge satellite data
    df_train_sat_2022 = pivot(pd.read_csv("output/satellite_train_2022.csv"))
    df_train_2022 = df_train_2022.merge(df_train_sat_2022, on=DESIGN_COLS, how="left")
    df_sat_2023 = pivot(pd.read_csv("output/satellite_train_2023.csv"))
    df_2023 = df_2023.merge(df_sat_2023, on=DESIGN_COLS, how="left")
    df_test_sat = pivot(pd.read_csv("output/satellite_validation_2023.csv"))
    df_test = df_test.merge(df_test_sat, on=DESIGN_COLS, how="left")

    # fix timepoints for each location
    # the comment is the difference of days between planting date and the chosen TP (i.e. 1, 2, or 3)
    VIS = [
        "_".join(x.split("_")[:2]) for x in df_train_sat_2022 if x not in DESIGN_COLS
    ]
    VIS = [x for x in VIS if "NDVI_max" not in x]  # I guess it has outliers?
    # VIS = [x for x in VIS if "EVI" not in x]
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
        "location",
        "experiment",
        "range",
        "row",
        "year",
        "genotype",
        "nitrogenTreatment",
        "img_id",
    ]

    FEATURES = [*CAT_COLS, *SAT_COLS]

    train = df_train_2022[["yieldPerAcre"] + FEATURES]
    val = df_2023[["yieldPerAcre"] + FEATURES]
    test = df_test[FEATURES]

    # checking correlations
    corrt = train.drop(CAT_COLS, axis=1).corr()["yieldPerAcre"].rename("train")
    corrv = val.drop(CAT_COLS, axis=1).corr()["yieldPerAcre"].rename("val")
    corr = pd.concat([corrt, corrv], axis=1).sort_values("train")
    corr["abs_diff"] = (corr["train"] - corr["val"]).abs()
    print(corr)

    # write
    train.to_csv("output/train.csv", index=False)
    val.to_csv("output/val.csv", index=False)
    test.to_csv("output/test.csv", index=False)

import pandas as pd

from constants import *


def convert_geno_name(x, mapping):
    if x in mapping.keys():
        x = mapping[x]
    return x


if __name__ == "__main__":
    geno_2022 = pd.read_csv(PATH_TRAIN_2022 / "HYBRID_HIPS_V3.5_ALLPLOTS.csv")[
        ["genotype"]
    ]
    geno_2023 = pd.read_csv(PATH_TRAIN_2023 / "train_HIPS_HYBRIDS_2023_V2.3.csv")[
        ["genotype"]
    ]
    geno_val_2023 = pd.read_csv(PATH_VAL / "val_HIPS_HYBRIDS_2023_V2.3.csv")[
        ["genotype"]
    ]
    geno = pd.concat([geno_2022, geno_2023, geno_val_2023])
    geno = geno.drop_duplicates(subset="genotype")

    # split parents
    geno[["parent1", "parent2"]] = geno["genotype"].str.split(" X ", expand=True)
    geno = geno[(geno["parent1"].notnull()) & (geno["parent2"].notnull())]
    geno = geno.drop("genotype", axis=1).reset_index(drop=True)

    # map genotype names to those of the genomic file
    mapping = pd.read_csv("output/geno_mapping.csv")
    mapping = dict(mapping[["genotype", "genotype_fix"]].values)
    geno["parent1"] = geno["parent1"].apply(lambda x: convert_geno_name(x, mapping))
    geno["parent2"] = geno["parent2"].apply(lambda x: convert_geno_name(x, mapping))
    geno.to_csv("output/hybrids.csv", index=False)
    
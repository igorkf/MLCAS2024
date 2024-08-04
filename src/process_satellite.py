from pathlib import Path
import argparse

import pandas as pd
import rasterio
import numpy as np
from tqdm import tqdm


def mask_array(arr):
    return np.ma.masked_array(arr, arr == 0)


def calc_stats(name, index):
    data = {
        f"{name}_mean": index.mean(),
        f"{name}_median": np.nanpercentile(np.ma.filled(index, np.nan), 0.5),
        f"{name}_min": index.min(),
        f"{name}_max": index.max(),
        f"{name}_sum": index.sum(),
        # f"{name}_std": index.std(),
    }
    return data


def create_keys(df):
    keys = ["location", "tp", "experiment", "range", "row"]
    df_keys = pd.DataFrame(
        df["file"]
        .str.split("-", expand=True)
        .apply(lambda x: [x[0], x[1], *x[2].split("_")], axis=1)
        .tolist(),
        columns=keys,
    )
    df = pd.concat([df_keys, df], axis=1)
    # df = df.drop("file", axis=1)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices={"train", "validation"}, required=True)
    parser.add_argument("--year", choices={"2022", "2023"}, required=True)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    PATH = Path(f"data/{args.data}/{args.year}")
    files = list(PATH.rglob("*.TIF"))
    if args.debug:
        np.random.seed(42)
        files = np.random.choice(files, size=20)
    data = []
    for file in tqdm(files):
        d = {}
        with rasterio.open(file) as src:
            r = mask_array(src.read(1).astype(float))
            g = mask_array(src.read(2).astype(float))
            # b = mask_array(src.read(3).astype(float))
            nir = mask_array(src.read(4).astype(float))
            re = mask_array(src.read(5).astype(float))
            db = src.read(6).astype(float)
            nir_minus_r = nir - r
            nir_plus_r = nir + r
            nir_minus_re = nir - re
            NDVI = nir_minus_r / nir_plus_r
            NDRE = nir_minus_re / (nir + re)
            MTCI = nir_minus_re / (re - r)
            CI = (nir / g) - 1
            # EVI = (2.5 * nir_minus_r) / (1 + nir + (6 * r) - (7.5 * b))
            # NGRDI = (r - g) / (g + r)
            # GNDVI = (nir - g) / (nir + g)
            # GLI = (2 * g - r - b) / (2 * g + r + b)
            # SAVI = 1.5 * nir_minus_r / (nir_plus_r + 0.5)
            d["path"] = str(file)
            d["file"] = file.stem
            d.update(calc_stats("NDVI", NDVI))
            d.update(calc_stats("NDRE", NDRE))
            d.update(calc_stats("MTCI", MTCI))
            d.update(calc_stats("CI", CI))
            # d.update(calc_stats("NGRDI", NGRDI))
            # d.update(calc_stats("GNDVI", GNDVI))
            # d.update(calc_stats("GLI", GLI))
            # d.update(calc_stats("SAVI", SAVI))
            data.append(d)
    df = pd.DataFrame(data)
    df = create_keys(df)
    df.to_csv(f"output/satellite_{args.data}_{args.year}.csv", index=False)

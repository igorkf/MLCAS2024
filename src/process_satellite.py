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
        f"{name}_std": index.std(),
    }
    return data


def create_keys(df):
    keys = ["location", "tp", "experiment", "range", "row"]
    df_keys = pd.DataFrame(
        df["id"]
        .str.split("-", expand=True)
        .apply(lambda x: [x[0], x[1], *x[2].split("_")], axis=1)
        .tolist(),
        columns=keys,
    )
    df = pd.concat([df_keys, df], axis=1)
    df = df.drop("id", axis=1)
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
            # g = src.read(2).astype(float)
            # b = src.read(3).astype(float)
            nir = mask_array(src.read(4).astype(float))
            re = mask_array(src.read(5).astype(float))
            # db = src.read(6).astype(float)
            NDVI = (nir - r) / (nir + r)
            NDRE = (nir - re) / (nir + re)
            d["id"] = file.stem
            d.update(calc_stats("NDVI", NDVI))
            d.update(calc_stats("NDRE", NDRE))
            data.append(d)
    df = pd.DataFrame(data)
    df = create_keys(df)
    df.to_csv(f"output/satellite_{args.data}_{args.year}.csv", index=False)

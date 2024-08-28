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
        f"{name}_sum": index.sum()
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
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices={"train", "validation", "test"}, required=True)
    parser.add_argument("--year", choices={"2022", "2023"}, required=True)
    parser.add_argument("--aggregate", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.data == "test":
        PATH = Path(f"data/test/Test/Test")
    else:
        PATH = Path(f"data/{args.data}/{args.year}/{args.year}")
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
            if args.aggregate:
                d.update(calc_stats("NDVI", NDVI))
                d.update(calc_stats("NDRE", NDRE))
                d.update(calc_stats("MTCI", MTCI))
                d.update(calc_stats("CI", CI))
            else:
                d["NDVI"] = NDVI[~r.mask].data.ravel()
                d["NDRE"] = NDRE[~r.mask].data.ravel()
                d["MTCI"] = MTCI[~r.mask].data.ravel()
                d["CI"] = CI[~r.mask].data.ravel()
            d["path"] = str(file)
            d["file"] = file.stem
            data.append(d)

    vis = ["NDVI", "NDRE", "MTCI", "CI"]
    df = pd.DataFrame(data)
    df = create_keys(df)
    if not args.aggregate:
        df = df.explode(vis, ignore_index=True)
        df[vis] = df[vis].astype(float)
        out = f"output/satellite_{args.data}_{args.year}_raw.csv"
    else:
        out = f"output/satellite_{args.data}_{args.year}.csv"
    print(df.isnull().sum() / len(df))
    df.to_csv(out, index=False)

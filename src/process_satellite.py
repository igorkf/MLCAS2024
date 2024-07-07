from pathlib import Path
import argparse

import pandas as pd
import rasterio
import numpy as np
from tqdm import tqdm


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


def pivot(df):
    df = df.pivot(index=['location', 'experiment', 'range', 'row'], columns=['tp'])
    df.columns = [f'{x[0]}_{x[1]}' for x in df.columns]
    df = df.reset_index()
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
        with rasterio.open(file) as src:
            r = src.read(1).astype(float)
            r = np.ma.masked_array(r, r == 0)  # to disconsider zero pixels
            # g = src.read(2).astype(float)
            # b = src.read(3).astype(float)
            nir = src.read(4).astype(float)
            # re = src.read(5).astype(float)
            # db = src.read(6).astype(float)
            NDVI = (nir - r) / (nir + r)
            d = {
                "id": file.stem,
                "NDVI_mean": NDVI.mean(),
                "NDVI_median": np.nanpercentile(np.ma.filled(NDVI, np.nan), 0.5),
                "NDVI_min": NDVI.min(),
                "NDVI_max": NDVI.max(),
            }
            data.append(d)
    df = pd.DataFrame(data)
    df = create_keys(df)
    df = pivot(df)
    df.to_csv(f"output/satellite_{args.data}_{args.year}.csv", index=False)

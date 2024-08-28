import pandas as pd
from sklearn.decomposition import PCA


def convert_geno_name(x, mapping):
    if x in mapping.keys():
        x = mapping[x]
    return x


N = 46
N_COMPONENTS = 45
cols = [f"pc{i + 1}" for i in range(N_COMPONENTS)]

if __name__ == "__main__":
    dtype = dict(
        zip(range(5 + N), ["str" for _ in range(5)] + ["int8" for _ in range(N)])
    )
    snps = pd.read_table("maize_numeric.txt", dtype=dtype)
    samples = snps.columns[5:].tolist()

    pca = PCA(n_components=N_COMPONENTS)
    df = pd.DataFrame(pca.fit_transform(snps.iloc[:, 5:].T + 1), columns=cols, index=samples)
    df = df.reset_index().rename(columns={"index": "genotype"})
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())
    
    mapping = pd.read_csv("output/geno_mapping.csv")
    mapping = dict(mapping[["genotype", "genotype_fix"]].values)
    df["genotype"] = df["genotype"].apply(lambda x: convert_geno_name(x, mapping))
    df.to_csv("output/pca_snps.csv", index=False)

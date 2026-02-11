# Process mQTL data according to Finucane lab finemapping protocol

import pandas as pd

data = pd.read_csv("../data/godmc/assoc_meta_all.csv.gz")

print(f"There are {len(data)} total SNP:CpG pairs")

# Identify and filter out CpGs with no significant associations

data["significant"] = data.groupby("cpg")["pval"].transform(lambda x: (x < 5e-8).any())

print(
    f"Proportion of CpGs with at least one significant SNP: {data.groupby('cpg')['significant'].first().sum() / len(data['cpg'].unique()):.4%}"
)
print(f"Total associations with significant CpGs: {data['significant'].sum()}")
data[["cpg", "snp", "pval", "significant"]].head(20)

data = data[data["significant"]]

print(f"There are {len(data)} SNP:CpG pairs after filtering to significant CpGs")

print("Filtering out MHC...")

data["pos"] = data["snp"].str.split(":").str[1].astype(int)
# Filter out MHC region
data = data[~((data["snp"].str.startswith("chr6")) & data["pos"].between(2.5e7, 3.6e7))]

print(f"There are {len(data)} SNP:CpG pairs after filtering out MHC")

print("Filtering out indels...")
data = data[data["snp"].str.endswith(":SNP")]
print(f"There are {len(data)} SNP:CpG pairs after filtering out indels")

print("Calculating Z-scores and formatting...")
data["chr"] = data["snp"].str.split(":").str[0].str.replace("chr", "")
# Remap numeric sex chromosomes to GRCh37 format (X, Y, MT)
data["chr"] = data["chr"].replace({"23": "X", "24": "Y", "25": "MT"})
data["Z"] = data["beta_a1"] / data["se"]

# Remove unused columns here for quicker loading
cols_to_drop = [
    "hetisq",
    "hetchisq",
    "hetpval",
    "tausq",
    "beta_are_a1",
    "se_are",
    "pval_are",
    "se_mre",
    "pval_mre",
]
data = data.drop(columns=cols_to_drop)

print("Saving processed data...")
data.to_csv("../data/godmc/assoc_meta_for_finemapping.csv", index=False)

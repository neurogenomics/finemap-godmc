# Prepare Z scores and LD matrices for finemapping.

import hail as hl
from hail.linalg import BlockMatrix
import os
import subprocess
import pandas as pd
import numpy as np
import argparse
import time
import socket

def match_variants(ht_snp: hl.Table, ht_idx: hl.Table) -> hl.Table:
    """
    Matches the variants in the GoDMC dataset with the reference panel.
    If there is no match initially, repeats for unmatched SNPs with flipped alleles to ensure maximum coverage.
    """
    # Match forward strand variants
    matched_forward = ht_snp.join(ht_idx, how="inner")
    matched_forward = matched_forward.annotate(flipped=False)

    # Find unmatched variants
    unmatched = ht_snp.anti_join(ht_idx)

    # Flip unmatched variants
    unmatched_flipped = unmatched.key_by(
        locus=unmatched.locus, alleles=hl.array([unmatched.allele1, unmatched.allele2])
    )

    # Match flipped variants
    matched_flipped = unmatched_flipped.join(ht_idx, how="inner")
    matched_flipped = matched_flipped.annotate(
        flipped=True,
        allele1=matched_flipped.allele2,
        allele2=matched_flipped.allele1,
    )

    # Combine all matches (counts printed after to_pandas for efficiency)
    ht_matched = matched_forward.union(matched_flipped)

    return ht_matched


def process_cpg(
    cpg_id: str, data: pd.DataFrame, bm: BlockMatrix, ht_idx: hl.Table, out_dir: str
):
    """Prepare Z scores and LD matrix for a single CpG site."""
    start = time.time()
    print(f"Processing CpG {cpg_id}...")

    subset = data[data["cpg"] == cpg_id]

    if len(subset) == 0:
        print(f"WARNING: No data found for CpG {cpg_id}, skipping.")
        return False

    # Get the lead SNP (smallest p-value)
    lead_snp = subset.loc[subset["pval"].idxmin()]
    snp_loc = lead_snp["pos"]

    # Extract all other SNPs within a 3Mb window on the same chromosome
    window_start = snp_loc - 1_500_000
    window_end = snp_loc + 1_500_000
    snp_df = subset[
        (subset["pos"].between(window_start, window_end))
        & (subset["chr"] == lead_snp["chr"])
    ].copy()

    snp_df["chr"] = snp_df["chr"].astype(str)

    # Convert SNPs to hail format
    ht_snp = hl.Table.from_pandas(snp_df)
    ht_snp = ht_snp.annotate(
        locus=hl.locus(ht_snp.chr, ht_snp.pos),
        alleles=hl.array(
            [ht_snp.allele2, ht_snp.allele1]
        ),  # In GoDMC allele2 is ref, allele1 is alt.
    )
    ht_snp = ht_snp.key_by(locus=ht_snp.locus, alleles=ht_snp.alleles)

    # Filter variant indices with CpG SNP data
    ht_matched = match_variants(ht_snp, ht_idx)

    # Order by idx and convert to pandas
    ht_matched = ht_matched.order_by(ht_matched.idx)
    final_snp_df = ht_matched.to_pandas()

    # Print counts
    n_forward = (~final_snp_df["flipped"]).sum()
    n_flipped = (final_snp_df["flipped"]).sum()
    print(
        f"  Matched {len(final_snp_df)} variants ({n_forward} forward, {n_flipped} flipped)"
    )

    # Flip signs and save csv
    final_snp_df["Z"] = np.where(
        final_snp_df["flipped"], -final_snp_df["Z"], final_snp_df["Z"]
    )
    final_snp_df["AF"] = np.where(
        final_snp_df["flipped"], 1 - final_snp_df["AF"], final_snp_df["AF"]
    )
    final_snp_df.to_csv(f"{out_dir}/{cpg_id}.csv", index=False)

    # Get indices
    idx = final_snp_df["idx"].tolist()
    bm_filtered = bm.filter(idx, idx)

    # Symmetrise and save the LD matrix
    ld_np = bm_filtered.to_numpy()
    ld_np = (ld_np + ld_np.T) / 2
    np.savetxt(f"{out_dir}/{cpg_id}_LD.txt", ld_np, delimiter=",")

    # Clean up intermediate tables
    ht_snp.unpersist()
    ht_matched.unpersist()

    print(f"  Data prep completed in {time.time() - start:.2f}s")
    return True


def run_susie(cpg_id: str, data_dir: str, out_dir: str, cleanup: bool = True):
    """Run SuSiE fine-mapping for a single CpG site."""
    start = time.time()
    print(f"  Running SuSiE for {cpg_id}...")

    csv_file = f"{data_dir}/{cpg_id}.csv"
    ld_file = f"{data_dir}/{cpg_id}_LD.txt"

    # Check input files exist
    if not os.path.exists(csv_file) or not os.path.exists(ld_file):
        print(f"  ERROR: Input files not found for {cpg_id}")
        return False

    try:
        result = subprocess.run(
            [
                "Rscript",
                "finemap_cpg.R",
                "--cpg",
                cpg_id,
                "--data-dir",
                data_dir + "/",
                "--out-dir",
                out_dir + "/",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  SuSiE completed in {time.time() - start:.2f}s")

        # Clean up intermediate files if requested
        if cleanup:
            os.remove(csv_file)
            os.remove(ld_file)
            print("  Cleaned up intermediate files")

        return True
    except subprocess.CalledProcessError as e:
        print(f"  SuSiE ERROR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Process data for finemapping (batched)"
    )
    parser.add_argument(
        "--cpg-list",
        required=True,
        help="Comma-separated list of CpG IDs or path to file with one CpG per line",
    )
    parser.add_argument(
        "--qtl-path", default="../data/godmc/assoc_meta_for_finemapping.csv"
    )
    parser.add_argument("--output-dir", default="../data/finemapping_tmp/")
    parser.add_argument(
        "--log-dir", default="../logs/hail/", help="Directory for Hail logs"
    )
    parser.add_argument(
        "--run-susie", action="store_true", help="Whether to run SuSiE after data prep"
    )
    parser.add_argument(
        "--susie-out-dir",
        default="../data/susie_results/",
        help="Output directory for SuSiE results",
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Whether to cleanup individual files"
    )

    args = parser.parse_args()
    qtl_path = args.qtl_path
    out_dir = args.output_dir

    # Ensure output directories exist
    os.makedirs(out_dir, exist_ok=True)
    if args.run_susie:
        os.makedirs(args.susie_out_dir, exist_ok=True)
    out_dir = args.output_dir

    # Parse CpG list
    if os.path.exists(args.cpg_list):
        # Process file path
        with open(args.cpg_list, "r") as f:
            cpg_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(cpg_ids)} CpG IDs from {args.cpg_list}")
    else:
        # Process comma-separated list
        cpg_ids = [cpg.strip() for cpg in args.cpg_list.split(",")]
        print(f"Processing {len(cpg_ids)} CpG IDs")

    os.environ["HAIL_LOG_DIR"] = args.log_dir

    print("Initializing Hail...")
    hl.init(
        master="local[*]",
        spark_conf={
            "spark.jars": "/rds/general/user/tc1125/home/jars/hadoop-aws-3.3.4.jar,/rds/general/user/tc1125/home/jars/aws-java-sdk-bundle-1.12.539.jar",
            "spark.hadoop.fs.s3a.connection.maximum": "200",
            "spark.hadoop.fs.s3a.connection.timeout": "600000",
            "spark.hadoop.fs.s3a.connection.establish.timeout": "600000",
            "spark.hadoop.fs.s3a.attempts.maximum": "10",
            "spark.network.timeout": "600s",
            "spark.driver.bindAddress": "0.0.0.0",
            "spark.driver.host": socket.gethostname(),
        }
    )

    # Load QTL data
    print(f"Loading QTL data from {qtl_path}...")
    data = pd.read_csv(qtl_path)

    # Load LD reference data
    print("Loading LD reference data from S3...")
    ld_matrix_path = "s3a://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.bm"
    ld_variant_index_path = (
        "s3a://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.variant.ht"
    )

    bm = BlockMatrix.read(ld_matrix_path)
    ht_idx = hl.read_table(ld_variant_index_path)
    print("LD reference data loaded")

    # Process each CpG
    print(f"\nProcessing {len(cpg_ids)} CpGs...\n")
    for i, cpg_id in enumerate(cpg_ids, 1):
        print(f"[{i}/{len(cpg_ids)}] ", end="")
        try:
            success = process_cpg(cpg_id, data, bm, ht_idx, out_dir)

            if success and args.run_susie:
                run_susie(cpg_id, out_dir, args.susie_out_dir, args.cleanup)
        except Exception as e:
            print(f"  ERROR processing {cpg_id}: {e}")
            continue

    print("\nFinished processing CpGs.")


if __name__ == "__main__":
    main()

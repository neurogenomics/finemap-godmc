# Prepare Z scores and LD matrices for finemapping - PROFILED VERSION

import hail as hl
from hail.linalg import BlockMatrix
import os
import pandas as pd
import numpy as np
import argparse
import time
from contextlib import contextmanager


@contextmanager
def timer(name: str, timings: dict):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    timings[name] = elapsed
    print(f"  [{name}] {elapsed:.2f}s")


def match_variants_profiled(
    ht_snp: hl.Table, ht_idx: hl.Table, timings: dict
) -> hl.Table:
    """
    Matches the variants in the GoDMC dataset with the reference panel.
    If there is no match initially, repeats for unmatched SNPs with flipped alleles to ensure maximum coverage.
    """
    with timer("join_forward", timings):
        matched_forward = ht_snp.join(ht_idx, how="inner")
        matched_forward = matched_forward.annotate(flipped=False)

    with timer("count_forward", timings):
        forward_count = matched_forward.count()
        print(f"    {forward_count} matches on forward strand")

    with timer("anti_join_unmatched", timings):
        unmatched = ht_snp.anti_join(ht_idx)

    with timer("count_unmatched", timings):
        unmatched_count = unmatched.count()
        print(f"    {unmatched_count} unmatched variants")

    with timer("flip_alleles", timings):
        unmatched_flipped = unmatched.key_by(
            locus=unmatched.locus,
            alleles=hl.array([unmatched.allele1, unmatched.allele2]),
        )

    with timer("join_flipped", timings):
        matched_flipped = unmatched_flipped.join(ht_idx, how="inner")
        matched_flipped = matched_flipped.annotate(
            flipped=True,
            allele1=matched_flipped.allele2,
            allele2=matched_flipped.allele1,
        )

    with timer("count_flipped", timings):
        flipped_count = matched_flipped.count()
        print(f"    {flipped_count} matches after flipping")

    with timer("union_matches", timings):
        ht_matched = matched_forward.union(matched_flipped)

    with timer("count_total", timings):
        total_count = ht_matched.count()
        print(f"    Total matched variants: {total_count}")

    return ht_matched


def process_cpg_profiled(
    cpg_id: str, data: pd.DataFrame, bm: BlockMatrix, ht_idx: hl.Table, out_dir: str
) -> dict:
    """Process a single CpG site with detailed profiling."""
    timings = {}
    total_start = time.time()
    print(f"Processing CpG {cpg_id}...")

    with timer("pandas_filter_subset", timings):
        subset = data[data["cpg"] == cpg_id]

    if len(subset) == 0:
        print(f"  WARNING: No data found for CpG {cpg_id}, skipping.")
        return timings

    with timer("pandas_find_lead_snp", timings):
        lead_snp = subset.loc[subset["pval"].idxmin()]
        snp_loc = lead_snp["pos"]

    with timer("pandas_window_filter", timings):
        window_start = snp_loc - 1_500_000
        window_end = snp_loc + 1_500_000
        snp_df = subset[
            (subset["pos"].between(window_start, window_end))
            & (subset["chr"] == lead_snp["chr"])
        ].copy()
        snp_df["chr"] = snp_df["chr"].astype(str)

    print(f"  SNPs in window: {len(snp_df)}")

    with timer("hail_from_pandas", timings):
        ht_snp = hl.Table.from_pandas(snp_df)

    with timer("hail_annotate_locus", timings):
        ht_snp = ht_snp.annotate(
            locus=hl.locus(ht_snp.chr, ht_snp.pos),
            alleles=hl.array([ht_snp.allele2, ht_snp.allele1]),
        )
        ht_snp = ht_snp.key_by(locus=ht_snp.locus, alleles=ht_snp.alleles)

    print("  Matching variants...")
    ht_matched = match_variants_profiled(ht_snp, ht_idx, timings)

    with timer("join_idx_filtered", timings):
        ht_idx_filtered = ht_idx.join(ht_matched.select(), how="inner")

    with timer("order_by_idx_filtered", timings):
        ht_idx_filtered = ht_idx_filtered.order_by(ht_idx_filtered.idx)

    with timer("order_by_matched", timings):
        ht_matched = ht_matched.order_by(ht_matched.idx)

    with timer("to_pandas", timings):
        final_snp_df = ht_matched.to_pandas()

    with timer("flip_signs", timings):
        final_snp_df["Z"] = np.where(
            final_snp_df["flipped"], -final_snp_df["Z"], final_snp_df["Z"]
        )
        final_snp_df["AF"] = np.where(
            final_snp_df["flipped"], 1 - final_snp_df["AF"], final_snp_df["AF"]
        )

    with timer("save_csv", timings):
        final_snp_df.to_csv(f"{out_dir}/{cpg_id}.csv", index=False)

    with timer("collect_idx", timings):
        idx = ht_idx_filtered.idx.collect()

    print(f"  Filtering BlockMatrix for {len(idx)} variants...")
    with timer("blockmatrix_filter", timings):
        bm_filtered = bm.filter(idx, idx)

    with timer("blockmatrix_to_numpy", timings):
        ld_np = bm_filtered.to_numpy()

    with timer("symmetrise_matrix", timings):
        ld_np = (ld_np + ld_np.T) / 2

    with timer("save_ld_matrix", timings):
        np.savetxt(f"{out_dir}/{cpg_id}_LD.txt", ld_np, delimiter=",")

    with timer("unpersist_cleanup", timings):
        ht_snp.unpersist()
        ht_matched.unpersist()
        ht_idx_filtered.unpersist()

    total_time = time.time() - total_start
    timings["TOTAL"] = total_time

    print(f"\n  === Timing Summary for {cpg_id} ===")
    sorted_timings = sorted(timings.items(), key=lambda x: -x[1])
    for name, elapsed in sorted_timings:
        pct = (elapsed / total_time) * 100
        print(f"  {name:30s} {elapsed:8.2f}s ({pct:5.1f}%)")
    print()

    return timings


def main():
    parser = argparse.ArgumentParser(
        description="Process data for finemapping (PROFILED)"
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

    args = parser.parse_args()
    qtl_path = args.qtl_path
    out_dir = args.output_dir

    # Parse CpG list
    if os.path.exists(args.cpg_list):
        with open(args.cpg_list, "r") as f:
            cpg_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(cpg_ids)} CpG IDs from {args.cpg_list}")
    else:
        cpg_ids = [cpg.strip() for cpg in args.cpg_list.split(",")]
        print(f"Processing {len(cpg_ids)} CpG IDs")

    os.environ["HAIL_LOG_DIR"] = args.log_dir

    print("Initializing Hail...")
    init_start = time.time()
    hl.init(
        spark_conf={
            "spark.jars": "/home/tobyc/data/jars/hadoop-aws-3.3.4.jar,/home/tobyc/data/jars/aws-java-sdk-bundle-1.12.539.jar",
            "spark.hadoop.fs.s3a.connection.maximum": "200",
            "spark.hadoop.fs.s3a.connection.timeout": "600000",
            "spark.hadoop.fs.s3a.connection.establish.timeout": "600000",
            "spark.hadoop.fs.s3a.attempts.maximum": "10",
            "spark.network.timeout": "600s",
        }
    )
    print(f"Hail initialized in {time.time() - init_start:.2f}s")

    print(f"Loading QTL data from {qtl_path}...")
    qtl_start = time.time()
    data = pd.read_csv(qtl_path)
    print(f"Loaded {len(data)} rows in {time.time() - qtl_start:.2f}s")

    print("Loading LD reference data from S3...")
    ld_start = time.time()
    ld_matrix_path = "s3a://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.bm"
    ld_variant_index_path = (
        "s3a://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.variant.ht"
    )

    bm = BlockMatrix.read(ld_matrix_path)
    ht_idx = hl.read_table(ld_variant_index_path)
    print(f"LD reference data loaded in {time.time() - ld_start:.2f}s")

    # Process each CpG and collect timings
    all_timings = []
    print(f"\n{'=' * 60}")
    print(f"Processing {len(cpg_ids)} CpGs with profiling...")
    print(f"{'=' * 60}\n")

    for i, cpg_id in enumerate(cpg_ids, 1):
        print(f"[{i}/{len(cpg_ids)}] ", end="")
        try:
            timings = process_cpg_profiled(cpg_id, data, bm, ht_idx, out_dir)
            timings["cpg_id"] = cpg_id
            all_timings.append(timings)
        except Exception as e:
            print(f"  ERROR processing {cpg_id}: {e}")
            continue

    # Print aggregate summary
    if all_timings:
        print(f"\n{'=' * 60}")
        print("AGGREGATE TIMING SUMMARY")
        print(f"{'=' * 60}")

        # Get all timing keys (excluding cpg_id)
        all_keys = set()
        for t in all_timings:
            all_keys.update(k for k in t.keys() if k != "cpg_id")

        # Calculate averages
        avg_timings = {}
        for key in all_keys:
            values = [t.get(key, 0) for t in all_timings]
            avg_timings[key] = sum(values) / len(values)

        total_avg = avg_timings.get("TOTAL", 1)
        sorted_avg = sorted(avg_timings.items(), key=lambda x: -x[1])

        print(f"\nAverage time per CpG (n={len(all_timings)}):")
        for name, elapsed in sorted_avg:
            pct = (elapsed / total_avg) * 100
            print(f"  {name:30s} {elapsed:8.2f}s ({pct:5.1f}%)")

    print(f"\nFinished processing all {len(cpg_ids)} CpGs.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Split QTL data by CpG ID into individual CSV files.

This preprocessing step avoids broadcasting large DataFrames to Spark executors
by creating small, per-CpG files that can be read individually.
"""

import pandas as pd
import os
import argparse
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Tuple, Optional


def save_cpg(args_tuple: Tuple) -> Tuple[str, bool, Optional[str]]:
    """
    Worker function to save a single CpG's data to CSV.
    
    Returns:
        Tuple of (cpg_id, success, error_message)
    """
    cpg_id, subset, output_dir = args_tuple
    output_file = os.path.join(output_dir, f"{cpg_id}.csv")
    try:
        subset.to_csv(output_file, index=False)
        return cpg_id, True, None
    except Exception as exc:
        return cpg_id, False, str(exc)


def main():
    parser = argparse.ArgumentParser(
        description="Split QTL association data into per-CpG files (parallel)"
    )
    parser.add_argument(
        "--qtl-path",
        required=True,
        help="Path to the QTL association data CSV file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save per-CpG CSV files",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(8, cpu_count()),
        help=f"Number of parallel workers (default: min(8, {cpu_count()}))",
    )
    args = parser.parse_args()

    # Use absolute path for clarity
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"\nLoading QTL data from {args.qtl_path}...")
    start = time.time()
    data = pd.read_csv(args.qtl_path)
    print(f"Loaded {len(data):,} rows in {time.time() - start:.1f}s")

    if len(data) == 0:
        print("ERROR: QTL file is empty. No CpGs to process.")
        return

    if "cpg" not in data.columns:
        print(f"ERROR: 'cpg' column not found. Available columns: {list(data.columns)}")
        return

    cpg_ids = set(data["cpg"].unique())
    print(f"Found {len(cpg_ids)} unique CpGs in data")

    # Prepare data for parallel processing
    print(f"\nPreparing data for {args.n_jobs} parallel workers...")
    groups = [(cpg_id, subset) for cpg_id, subset in data.groupby("cpg")]

    # Add output_dir to each tuple for the worker
    work_items = [(cpg_id, subset, output_dir) for cpg_id, subset in groups]

    print(f"\nWriting {len(work_items)} CpG files...")
    start = time.time()

    with Pool(processes=args.n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap(save_cpg, work_items),
                desc="Saving CpG files",
                total=len(work_items),
            )
        )

    # Check for failures
    failures = [(cpg_id, err) for cpg_id, ok, err in results if not ok]
    if failures:
        print(f"\nWARNING: {len(failures)} CpGs failed to save:")
        for cpg_id, err in failures[:10]:
            print(f"  {cpg_id}: {err}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    # Verify output
    written_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
    total_time = time.time() - start

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Saved: {len(results) - len(failures)} CpGs")
    print(f"  Failed: {len(failures)}")
    print(f"  CSV files in output: {len(written_files)}")
    print(f"  Workers: {args.n_jobs}")
    print(f"  Output directory: {output_dir}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

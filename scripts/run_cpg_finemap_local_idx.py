#!/usr/bin/env python
# Prepare Z scores and LD matrices for finemapping with local variant index cache.

import argparse
import os
import shutil
import subprocess
import time
from typing import Tuple

import hail as hl
import numpy as np
import pandas as pd
from hail.linalg import BlockMatrix


TRANSIENT_ERROR_PATTERNS = (
    "Timeout waiting for connection from pool",
    "ConnectionPoolTimeoutException",
    "InterruptedIOException",
    "TaskKilled (Stage cancelled",
    "Job aborted due to stage failure",
)


def match_variants(ht_snp: hl.Table, ht_idx: hl.Table) -> hl.Table:
    """
    Match variants in the CpG data with the Pan-UKB variant index.

    First match forward strand variants, then retry unmatched variants with
    flipped alleles for additional recovery.
    """
    matched_forward = ht_snp.join(ht_idx, how="inner")
    matched_forward = matched_forward.annotate(flipped=False)

    unmatched = ht_snp.anti_join(ht_idx)
    unmatched_flipped = unmatched.key_by(
        locus=unmatched.locus, alleles=hl.array([unmatched.allele1, unmatched.allele2])
    )

    matched_flipped = unmatched_flipped.join(ht_idx, how="inner")
    matched_flipped = matched_flipped.annotate(
        flipped=True,
        allele1=matched_flipped.allele2,
        allele2=matched_flipped.allele1,
    )

    return matched_forward.union(matched_flipped)


def is_transient_hail_error(err: Exception) -> bool:
    text = str(err)
    return any(pattern in text for pattern in TRANSIENT_ERROR_PATTERNS)


def acquire_lock(lock_dir: str, timeout_s: int, stale_lock_s: int, poll_s: int = 5) -> bool:
    """Acquire a directory lock using atomic mkdir."""
    start = time.time()
    while True:
        try:
            os.mkdir(lock_dir)
            with open(os.path.join(lock_dir, "owner.txt"), "w") as f:
                f.write(f"pid={os.getpid()}\n")
                f.write(f"host={os.uname().nodename}\n")
                f.write(f"created={int(time.time())}\n")
            return True
        except FileExistsError:
            # If lock looks stale, remove and retry
            try:
                lock_age = time.time() - os.path.getmtime(lock_dir)
                if lock_age > stale_lock_s:
                    print(
                        f"Cache lock appears stale ({lock_age:.0f}s old), removing: {lock_dir}"
                    )
                    shutil.rmtree(lock_dir, ignore_errors=True)
                    continue
            except FileNotFoundError:
                # Race condition: lock removed between checks
                continue

            if time.time() - start > timeout_s:
                return False
            time.sleep(poll_s)


def release_lock(lock_dir: str) -> None:
    shutil.rmtree(lock_dir, ignore_errors=True)


def ensure_local_variant_index(
    s3_variant_index_path: str,
    variant_cache_dir: str,
    refresh_cache: bool,
    wait_timeout_s: int,
    stale_lock_s: int,
) -> str:
    """
    Ensure a shared per-node local copy of UKBB.EUR.ldadj.variant.ht exists.

    Uses a lock directory to avoid concurrent copy attempts from multiple jobs.
    """
    os.makedirs(variant_cache_dir, exist_ok=True)
    local_variant_ht = os.path.join(variant_cache_dir, "UKBB.EUR.ldadj.variant.ht")
    lock_dir = os.path.join(variant_cache_dir, ".variant_ht.lock")

    if os.path.exists(local_variant_ht) and not refresh_cache:
        print(f"Using existing local variant index cache: {local_variant_ht}")
        return local_variant_ht

    print(f"Checking local variant index cache at: {local_variant_ht}")
    got_lock = acquire_lock(lock_dir, timeout_s=wait_timeout_s, stale_lock_s=stale_lock_s)
    if not got_lock:
        raise RuntimeError(
            f"Timed out waiting for variant index cache lock after {wait_timeout_s}s: {lock_dir}"
        )

    try:
        # Re-check after lock acquisition
        if os.path.exists(local_variant_ht) and not refresh_cache:
            print(f"Variant index cache created by another job: {local_variant_ht}")
            return local_variant_ht

        if refresh_cache and os.path.exists(local_variant_ht):
            print(f"Refreshing variant index cache: removing {local_variant_ht}")
            shutil.rmtree(local_variant_ht, ignore_errors=True)

        tmp_variant_ht = (
            f"{local_variant_ht}.tmp_{os.getpid()}_{int(time.time())}"
        )
        if os.path.exists(tmp_variant_ht):
            shutil.rmtree(tmp_variant_ht, ignore_errors=True)

        print("Copying variant index from S3 to local cache (one-time per node)...")
        start = time.time()
        ht_variant = hl.read_table(s3_variant_index_path)
        ht_variant.write(tmp_variant_ht, overwrite=True)
        os.replace(tmp_variant_ht, local_variant_ht)
        print(f"Local variant index cache ready in {time.time() - start:.2f}s")

    finally:
        release_lock(lock_dir)

    return local_variant_ht


def process_cpg(
    cpg_id: str,
    cpg_data_dir: str,
    bm: BlockMatrix,
    ht_idx: hl.Table,
    out_dir: str,
) -> bool:
    """Prepare Z scores and LD matrix for one CpG."""
    start = time.time()
    print(f"Processing CpG {cpg_id}...")

    cpg_file = os.path.join(cpg_data_dir, f"{cpg_id}.csv")
    if not os.path.exists(cpg_file):
        for ext in (".csv.gz", ".csv.bz2", ".csv.zip", ".csv.xz"):
            alt_file = os.path.join(cpg_data_dir, f"{cpg_id}{ext}")
            if os.path.exists(alt_file):
                cpg_file = alt_file
                break

    if not os.path.exists(cpg_file):
        print(f"WARNING: No data file found for CpG {cpg_id}, skipping.")
        return False

    snp_df = pd.read_csv(cpg_file)
    if len(snp_df) == 0:
        print(f"WARNING: No data found for CpG {cpg_id}, skipping.")
        return False

    snp_df["chr"] = snp_df["chr"].astype(str)

    lead_snp = snp_df.loc[snp_df["pval"].idxmin()]
    lead_chr = str(lead_snp["chr"])
    snp_loc = lead_snp["pos"]

    window_start = snp_loc - 1_500_000
    window_end = snp_loc + 1_500_000
    snp_df = snp_df[
        (snp_df["pos"].between(window_start, window_end))
        & (snp_df["chr"] == lead_chr)
    ].copy()

    if len(snp_df) == 0:
        print(f"WARNING: No SNPs in 3Mb window for CpG {cpg_id}, skipping.")
        return False

    ht_snp = hl.Table.from_pandas(snp_df)
    ht_snp = ht_snp.annotate(
        locus=hl.locus(ht_snp.chr, ht_snp.pos),
        alleles=hl.array([ht_snp.allele2, ht_snp.allele1]),
    )
    ht_snp = ht_snp.key_by(locus=ht_snp.locus, alleles=ht_snp.alleles)

    ht_matched = match_variants(ht_snp, ht_idx)
    final_snp_df = ht_matched.to_pandas()

    if len(final_snp_df) == 0:
        print(f"WARNING: No matched variants for CpG {cpg_id}, skipping.")
        return False

    # Faster than distributed order_by for these per-CpG tables
    final_snp_df.sort_values("idx", inplace=True)

    n_forward = (~final_snp_df["flipped"]).sum()
    n_flipped = final_snp_df["flipped"].sum()
    print(f"  Matched {len(final_snp_df)} variants ({n_forward} forward, {n_flipped} flipped)")

    final_snp_df["Z"] = np.where(final_snp_df["flipped"], -final_snp_df["Z"], final_snp_df["Z"])
    final_snp_df["AF"] = np.where(final_snp_df["flipped"], 1 - final_snp_df["AF"], final_snp_df["AF"])
    final_snp_df.to_csv(os.path.join(out_dir, f"{cpg_id}.csv"), index=False)

    idx = final_snp_df["idx"].tolist()
    bm_filtered = bm.filter(idx, idx)

    ld_np = bm_filtered.to_numpy()
    ld_np = (ld_np + ld_np.T) / 2
    np.savetxt(os.path.join(out_dir, f"{cpg_id}_LD.txt"), ld_np, delimiter=",")

    print(f"  Data prep completed in {time.time() - start:.2f}s")
    return True


def run_susie(cpg_id: str, data_dir: str, out_dir: str, cleanup: bool = True) -> bool:
    """Run SuSiE fine-mapping for one CpG."""
    start = time.time()
    print(f"  Running SuSiE for {cpg_id}...")

    csv_file = os.path.join(data_dir, f"{cpg_id}.csv")
    ld_file = os.path.join(data_dir, f"{cpg_id}_LD.txt")

    if not os.path.exists(csv_file) or not os.path.exists(ld_file):
        print(f"  ERROR: Input files not found for {cpg_id}")
        return False

    try:
        subprocess.run(
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

        if cleanup:
            os.remove(csv_file)
            os.remove(ld_file)
            print("  Cleaned up intermediate files")

        return True

    except subprocess.CalledProcessError as err:
        print(f"  SuSiE ERROR: {err.stderr}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process data for finemapping with local shared variant index cache"
    )
    parser.add_argument(
        "--cpg-list",
        required=True,
        help="Comma-separated list of CpG IDs or path to file with one CpG per line",
    )
    parser.add_argument(
        "--cpg-data-dir",
        required=True,
        help="Directory containing per-CpG CSV files (from split_qtl_by_cpg.py)",
    )
    parser.add_argument("--output-dir", default="../data/finemapping_tmp/")
    parser.add_argument("--log-dir", default="../logs/hail/", help="Directory for Hail logs")
    parser.add_argument("--run-susie", action="store_true", help="Run SuSiE after data prep")
    parser.add_argument(
        "--susie-out-dir",
        default="../data/susie_results/",
        help="Output directory for SuSiE results",
    )
    parser.add_argument("--cleanup", action="store_true", help="Cleanup intermediate files")

    parser.add_argument(
        "--variant-cache-dir",
        default=None,
        help="Shared per-node directory for local variant.ht cache (default: $EPHEMERAL/pan_ukb_cache)",
    )
    parser.add_argument(
        "--refresh-variant-cache",
        action="store_true",
        help="Force rebuilding local variant.ht cache from S3",
    )
    parser.add_argument(
        "--cache-wait-timeout",
        type=int,
        default=1800,
        help="Seconds to wait for another job to finish building the local cache",
    )
    parser.add_argument(
        "--cache-lock-stale-seconds",
        type=int,
        default=7200,
        help="Treat cache lock as stale after this many seconds",
    )
    parser.add_argument(
        "--hail-cores",
        type=int,
        default=8,
        help="Number of local Spark cores for Hail (default: 8)",
    )
    parser.add_argument(
        "--max-cpg-retries",
        type=int,
        default=1,
        help="Retries for transient Hail/S3 failures per CpG",
    )

    args = parser.parse_args()

    out_dir = args.output_dir
    cpg_data_dir = args.cpg_data_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.run_susie:
        os.makedirs(args.susie_out_dir, exist_ok=True)

    if os.path.exists(args.cpg_list):
        with open(args.cpg_list, "r") as f:
            cpg_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(cpg_ids)} CpG IDs from {args.cpg_list}")
    else:
        cpg_ids = [cpg.strip() for cpg in args.cpg_list.split(",") if cpg.strip()]
        print(f"Processing {len(cpg_ids)} CpG IDs")

    if not os.path.exists(cpg_data_dir):
        print(f"ERROR: CpG data directory not found: {cpg_data_dir}")
        print("Run split_qtl_by_cpg.py first to create per-CpG files")
        return

    job_id = os.environ.get("PBS_ARRAY_INDEX", str(os.getpid()))
    job_log_dir = os.path.join(args.log_dir, f"job_{job_id}")
    os.makedirs(job_log_dir, exist_ok=True)
    os.environ["HAIL_LOG_DIR"] = job_log_dir

    ephemeral_base = os.environ.get("EPHEMERAL", os.path.dirname(os.path.abspath(out_dir)))
    tmp_dir = os.path.join(ephemeral_base, f"hail_tmp_{job_id}")
    os.makedirs(tmp_dir, exist_ok=True)

    variant_cache_dir = args.variant_cache_dir
    if variant_cache_dir is None:
        variant_cache_dir = os.path.join(ephemeral_base, "pan_ukb_cache")

    spark_conf = {
        "spark.jars": "/rds/general/user/tc1125/home/jars/hadoop-aws-3.3.4.jar,/rds/general/user/tc1125/home/jars/aws-java-sdk-bundle-1.12.539.jar",
        "spark.driver.memory": "16g",
        "spark.driver.maxResultSize": "8g",
        "spark.executor.memory": "16g",
        "spark.executor.cores": str(args.hail_cores),
        "spark.default.parallelism": str(args.hail_cores),
        "spark.sql.shuffle.partitions": str(args.hail_cores),
        "spark.hadoop.fs.s3a.connection.maximum": "200",
        "spark.hadoop.fs.s3a.connection.establish.timeout": "300000",
        "spark.hadoop.fs.s3a.connection.timeout": "300000",
        "spark.hadoop.fs.s3a.connection.acquisition.timeout": "300000",
        "spark.hadoop.fs.s3a.threads.max": "64",
        "spark.hadoop.fs.s3a.attempts.maximum": "20",
        "spark.hadoop.fs.s3a.retry.limit": "20",
        "spark.hadoop.fs.s3a.retry.interval": "1000ms",
        "spark.network.timeout": "600s",
        "spark.rpc.askTimeout": "600s",
        "spark.rpc.lookupTimeout": "600s",
        "spark.driver.bindAddress": "127.0.0.1",
        "spark.driver.host": "localhost",
        "spark.local.dir": tmp_dir,
        "spark.ui.enabled": "false",
        "spark.driver.port": "0",
        "spark.blockManager.port": "0",
        "spark.port.maxRetries": "50",
    }

    ld_matrix_path = "s3a://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.bm"
    ld_variant_index_path = "s3a://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.variant.ht"

    print(f"Using pre-split CpG data from: {cpg_data_dir}")
    print(f"Using per-node variant cache directory: {variant_cache_dir}")

    def init_hail() -> Tuple[BlockMatrix, hl.Table]:
        print("Initializing Hail...")
        hl.init(
            master=f"local[{args.hail_cores}]",
            tmp_dir=tmp_dir,
            spark_conf=spark_conf,
            min_block_size=256,
            idempotent=True,
        )

        local_variant_index = ensure_local_variant_index(
            s3_variant_index_path=ld_variant_index_path,
            variant_cache_dir=variant_cache_dir,
            refresh_cache=args.refresh_variant_cache,
            wait_timeout_s=args.cache_wait_timeout,
            stale_lock_s=args.cache_lock_stale_seconds,
        )

        print("Loading LD reference data (BM from S3, variant index from local cache)...")
        bm = BlockMatrix.read(ld_matrix_path)
        ht_idx = hl.read_table(local_variant_index)
        print("LD reference data loaded")
        return bm, ht_idx

    def stop_hail() -> None:
        print("Stopping Hail...")
        try:
            hl.stop()
        except Exception:
            pass

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            os.makedirs(tmp_dir, exist_ok=True)
        time.sleep(2)

    bm, ht_idx = init_hail()

    print(f"\nProcessing {len(cpg_ids)} CpGs...\n")
    for i, cpg_id in enumerate(cpg_ids, 1):
        susie_output_file = os.path.join(args.susie_out_dir, f"{cpg_id}_susie.csv")
        if args.run_susie and os.path.exists(susie_output_file):
            print(f"[{i}/{len(cpg_ids)}] Skipping {cpg_id} - already completed")
            continue

        print(f"[{i}/{len(cpg_ids)}] ", end="")
        attempt = 0
        while attempt <= args.max_cpg_retries:
            try:
                success = process_cpg(cpg_id, cpg_data_dir, bm, ht_idx, out_dir)
                if success and args.run_susie:
                    run_susie(cpg_id, out_dir, args.susie_out_dir, args.cleanup)
                break
            except Exception as err:
                is_transient = is_transient_hail_error(err)
                if is_transient and attempt < args.max_cpg_retries:
                    print(
                        f"  Transient Hail/S3 error for {cpg_id} (attempt {attempt + 1}/"
                        f"{args.max_cpg_retries + 1}): {err}"
                    )
                    print("  Reinitializing Hail and retrying...")
                    stop_hail()
                    bm, ht_idx = init_hail()
                    attempt += 1
                    continue

                print(f"  ERROR processing {cpg_id}: {err}")
                break

    stop_hail()
    print("\nFinished processing CpGs.")


if __name__ == "__main__":
    main()

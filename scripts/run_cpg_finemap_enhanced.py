#!/usr/bin/env python3
"""
run_cpg_finemap_enhanced.py

Enhanced fine-mapping with improved connection management:
- BlockMatrix persist/unpersist to prevent S3 re-reads
- Retry logic with exponential backoff for S3 errors
- Dynamic restart based on memory usage or CpG count
- Optimized Spark configuration (500 connection pool)
- Resume capability via SuSiE output check
"""

import hail as hl
from hail.linalg import BlockMatrix
import os
import subprocess
import shutil
import pandas as pd
import numpy as np
import argparse
import time
import random
import psutil
from functools import wraps
from typing import Optional, Tuple

# Constants
MAX_RETRIES = 5
BASE_DELAY = 10
RESTART_INTERVAL = 250  # Restart after this many CpGs
MEMORY_THRESHOLD = 80   # Restart if memory usage exceeds this %
LD_S3_BUCKET = "s3a://pan-ukb-us-east-1/ld_release"


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable S3/network error."""
    error_str = str(error).lower()
    retryable_patterns = [
        'connection pool exhausted',
        'socket timeout',
        'connection timeout',
        's3a',
        'connection reset',
        'broken pipe',
        'unable to execute http request',
        'connection refused',
        'premature end of content',
        'unexpected end of stream',
        'java.net.socketexception',
        'java.io.ioexception',
    ]
    return any(pattern in error_str for pattern in retryable_patterns)


def retry_with_backoff(max_retries: int = MAX_RETRIES, base_delay: int = BASE_DELAY):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not is_retryable_error(e):
                        raise
                    
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), 300)
                        jitter = random.uniform(0, delay * 0.1)
                        total_delay = delay + jitter
                        
                        print(f"    ⚠️  Retry {attempt + 1}/{max_retries} after error: {str(e)[:80]}...")
                        print(f"       Waiting {total_delay:.1f}s...")
                        time.sleep(total_delay)
                    else:
                        print(f"    ❌ All {max_retries} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator


def get_spark_conf(tmp_dir: str) -> dict:
    """Get optimized Spark configuration."""
    return {
        # Jars
        "spark.jars": "/rds/general/user/tc1125/home/jars/hadoop-aws-3.3.4.jar,/rds/general/user/tc1125/home/jars/aws-java-sdk-bundle-1.12.539.jar",
        
        # S3 Connection Pool - Optimized for many operations
        "spark.hadoop.fs.s3a.connection.maximum": "500",
        "spark.hadoop.fs.s3a.connection.pool.max": "500",
        "spark.hadoop.fs.s3a.connection.pool.size": "200",
        "spark.hadoop.fs.s3a.connection.pool.idle.time": "600000",
        "spark.hadoop.fs.s3a.connection.timeout": "1200000",
        "spark.hadoop.fs.s3a.connection.establish.timeout": "1200000",
        "spark.hadoop.fs.s3a.attempts.maximum": "30",
        "spark.hadoop.fs.s3a.retry.limit": "30",
        "spark.hadoop.fs.s3a.retry.interval": "1000ms",
        
        # S3 Performance
        "spark.hadoop.fs.s3a.readahead.range": "512K",
        "spark.hadoop.fs.s3a.input.fadvise": "sequential",
        "spark.hadoop.fs.s3a.threads.max": "128",
        
        # Spark Memory and Performance
        "spark.network.timeout": "1200s",
        "spark.executor.heartbeatInterval": "120s",
        "spark.storage.blockManagerSlaveTimeoutMs": "1200000",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        
        # Local settings
        "spark.driver.bindAddress": "127.0.0.1",
        "spark.driver.host": "localhost",
        "spark.local.dir": tmp_dir,
        "spark.ui.enabled": "false",
        "spark.driver.port": "0",
        "spark.blockManager.port": "0",
    }


def check_memory(memory_threshold: int) -> Tuple[bool, float]:
    """Check memory usage and return (ok, percent_used)."""
    memory = psutil.virtual_memory()
    return memory.percent < memory_threshold, memory.percent


def match_variants(ht_snp: hl.Table, ht_idx: hl.Table) -> hl.Table:
    """Match variants with optimized persistence."""
    # Match forward strand variants
    matched_forward = ht_snp.join(ht_idx, how="inner")
    matched_forward = matched_forward.annotate(flipped=False)
    
    # Find unmatched variants
    unmatched = ht_snp.anti_join(ht_idx)
    
    # Flip unmatched variants
    unmatched_flipped = unmatched.key_by(
        locus=unmatched.locus,
        alleles=hl.array([unmatched.allele1, unmatched.allele2])
    )
    
    # Match flipped variants
    matched_flipped = unmatched_flipped.join(ht_idx, how="inner")
    matched_flipped = matched_flipped.annotate(
        flipped=True,
        allele1=matched_flipped.allele2,
        allele2=matched_flipped.allele1,
    )
    
    # Combine all matches
    ht_matched = matched_forward.union(matched_flipped)
    
    # Persist to avoid recomputation
    ht_matched = ht_matched.persist()
    
    return ht_matched


@retry_with_backoff(max_retries=5, base_delay=10)
def process_cpg(
    cpg_id: str,
    data: pd.DataFrame,
    bm: BlockMatrix,
    ht_idx: hl.Table,
    out_dir: str
) -> bool:
    """Prepare Z scores and LD matrix for a single CpG site."""
    start = time.time()
    
    subset = data[data["cpg"] == cpg_id]
    
    if len(subset) == 0:
        print(f"    WARNING: No data found for CpG {cpg_id}, skipping.")
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
    
    # Convert SNPs to hail format
    ht_snp = hl.Table.from_pandas(snp_df)
    ht_snp = ht_snp.annotate(
        locus=hl.locus(ht_snp.chr, ht_snp.pos),
        alleles=hl.array([ht_snp.allele2, ht_snp.allele1]),
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
    print(f"    Matched {len(final_snp_df)} variants ({n_forward} forward, {n_flipped} flipped)")
    
    if len(final_snp_df) == 0:
        print(f"    WARNING: No matching variants for CpG {cpg_id}")
        return False
    
    # Flip signs and save csv
    final_snp_df["Z"] = np.where(
        final_snp_df["flipped"], -final_snp_df["Z"], final_snp_df["Z"]
    )
    final_snp_df["AF"] = np.where(
        final_snp_df["flipped"], 1 - final_snp_df["AF"], final_snp_df["AF"]
    )
    final_snp_df.to_csv(f"{out_dir}/{cpg_id}.csv", index=False)
    
    # Get indices and filter BlockMatrix
    idx = final_snp_df["idx"].tolist()
    
    # CRITICAL: Persist filtered matrix to avoid recomputation
    bm_filtered = bm.filter(idx, idx).cache()
    
    # Convert to numpy and symmetrize
    ld_np = bm_filtered.to_numpy()
    ld_np = (ld_np + ld_np.T) / 2
    np.savetxt(f"{out_dir}/{cpg_id}_LD.txt", ld_np, delimiter=",")
    
    # CRITICAL: Unpersist to free memory immediately
    bm_filtered.unpersist()
    
    # Clean up intermediate tables
    ht_snp.unpersist()
    ht_matched.unpersist()
    
    elapsed = time.time() - start
    print(f"    ✓ Completed in {elapsed:.2f}s")
    return True


def run_susie(cpg_id: str, data_dir: str, out_dir: str, cleanup: bool = True) -> bool:
    """Run SuSiE fine-mapping for a single CpG site."""
    start = time.time()
    
    csv_file = f"{data_dir}/{cpg_id}.csv"
    ld_file = f"{data_dir}/{cpg_id}_LD.txt"
    
    # Check input files exist
    if not os.path.exists(csv_file) or not os.path.exists(ld_file):
        print(f"    ERROR: Input files not found for {cpg_id}")
        return False
    
    try:
        result = subprocess.run(
            [
                "Rscript",
                "finemap_cpg.R",
                "--cpg", cpg_id,
                "--data-dir", data_dir + "/",
                "--out-dir", out_dir + "/",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        
        elapsed = time.time() - start
        print(f"    ✓ SuSiE completed in {elapsed:.2f}s")
        
        # Clean up intermediate files if requested
        if cleanup:
            os.remove(csv_file)
            os.remove(ld_file)
            print(f"    Cleaned up intermediate files")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: SuSiE failed: {e.stderr[:200]}")
        return False


def init_hail(tmp_dir: str, log_path: str) -> Tuple[BlockMatrix, hl.Table]:
    """Initialize Hail and load LD reference data."""
    print("  Initializing Hail...")
    
    spark_conf = get_spark_conf(tmp_dir)
    
    hl.init(
        master="local[*]",
        tmp_dir=tmp_dir,
        spark_conf=spark_conf,
        idempotent=True,
        log=log_path,
    )
    
    print("  Loading LD reference data from S3...")
    bm = BlockMatrix.read(f"{LD_S3_BUCKET}/UKBB.EUR.ldadj.bm")
    ht_idx = hl.read_table(f"{LD_S3_BUCKET}/UKBB.EUR.ldadj.variant.ht")
    
    # CRITICAL: Persist to avoid re-reading during processing
    print("  Persisting LD data...")
    bm = bm.persist(storage_level="MEMORY_AND_DISK")
    ht_idx = ht_idx.persist(storage_level="MEMORY_AND_DISK")
    
    print(f"  ✓ LD data loaded: {bm.shape[0]} variants")
    return bm, ht_idx


def stop_hail(tmp_dir: str):
    """Stop Hail and clean up."""
    print("  Stopping Hail...")
    hl.stop()
    
    # Clean up temp directory
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
    
    time.sleep(5)  # Allow connections to fully close
    print("  ✓ Hail stopped and cleaned up")


def should_restart(cpgs_since_restart: int, last_restart_time: float, restart_interval: int, memory_threshold: int) -> Tuple[bool, str]:
    """
    Determine if we should restart Hail based on multiple criteria.
    Returns (should_restart, reason).
    """
    # Check CpG count
    if cpgs_since_restart >= restart_interval:
        return True, f"reached {restart_interval} CpGs"
    
    # Check memory usage
    memory_ok, memory_pct = check_memory(memory_threshold)
    if not memory_ok:
        return True, f"memory usage at {memory_pct:.1f}%"
    
    # Check time since last restart (optional: restart every 4 hours)
    elapsed_hours = (time.time() - last_restart_time) / 3600
    if elapsed_hours >= 4:
        return True, f"4 hours since last restart"
    
    return False, ""


def main():
    parser = argparse.ArgumentParser(
        description="Process data for finemapping (enhanced with connection management)"
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
    parser.add_argument(
        "--restart-interval",
        type=int,
        default=RESTART_INTERVAL,
        help=f"Restart Hail after this many CpGs (default: {RESTART_INTERVAL})",
    )
    parser.add_argument(
        "--memory-threshold",
        type=int,
        default=MEMORY_THRESHOLD,
        help=f"Restart if memory usage exceeds this %% (default: {MEMORY_THRESHOLD})",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    qtl_path = args.qtl_path
    out_dir = args.output_dir
    
    os.makedirs(out_dir, exist_ok=True)
    if args.run_susie:
        os.makedirs(args.susie_out_dir, exist_ok=True)
    
    # Parse CpG list
    if os.path.exists(args.cpg_list):
        with open(args.cpg_list, "r") as f:
            cpg_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(cpg_ids)} CpG IDs from {args.cpg_list}")
    else:
        cpg_ids = [cpg.strip() for cpg in args.cpg_list.split(",")]
        print(f"Processing {len(cpg_ids)} CpG IDs")
    
    # Setup job-specific directories
    job_id = os.environ.get("PBS_ARRAY_INDEX", str(os.getpid()))
    job_log_dir = os.path.join(args.log_dir, f"job_{job_id}")
    os.makedirs(job_log_dir, exist_ok=True)
    
    ephemeral_base = os.environ.get("EPHEMERAL", os.path.dirname(os.path.abspath(out_dir)))
    tmp_dir = os.path.join(ephemeral_base, f"hail_tmp_{job_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Job {job_id}: Processing {len(cpg_ids)} CpGs")
    print(f"Restart interval: {RESTART_INTERVAL} CpGs")
    print(f"Memory threshold: {MEMORY_THRESHOLD}%")
    print(f"{'='*60}\n")
    
    # Load QTL data
    print(f"Loading QTL data from {qtl_path}...")
    data = pd.read_csv(qtl_path)
    print(f"Loaded {len(data)} QTL associations\n")
    
    # Initialize Hail for first batch
    bm, ht_idx = init_hail(tmp_dir, os.path.join(job_log_dir, "hail.log"))
    
    # Process CpGs
    total_processed = 0
    total_failed = 0
    cpgs_since_restart = 0
    last_restart_time = time.time()
    start_time = time.time()
    
    for i, cpg_id in enumerate(cpg_ids, 1):
        # Check if already completed (resume capability)
        susie_output_file = os.path.join(args.susie_out_dir, f"{cpg_id}_susie.csv")
        if os.path.exists(susie_output_file):
            print(f"[{i}/{len(cpg_ids)}] {cpg_id}: Already completed, skipping")
            total_processed += 1
            continue
        
        # Check if we should restart Hail
        should_restart_flag, restart_reason = should_restart(cpgs_since_restart, last_restart_time, args.restart_interval, args.memory_threshold)
        
        if should_restart_flag and i > 1:
            print(f"\n{'-'*60}")
            print(f"Restarting Hail: {restart_reason}")
            print(f"{'-'*60}\n")
            
            stop_hail(tmp_dir)
            bm, ht_idx = init_hail(tmp_dir, os.path.join(job_log_dir, f"hail_restart_{i}.log"))
            
            cpgs_since_restart = 0
            last_restart_time = time.time()
        
        # Process CpG
        print(f"\n[{i}/{len(cpg_ids)}] Processing {cpg_id}...")
        
        try:
            success = process_cpg(cpg_id, data, bm, ht_idx, out_dir)
            
            if success and args.run_susie:
                susie_success = run_susie(cpg_id, out_dir, args.susie_out_dir, args.cleanup)
                if susie_success:
                    total_processed += 1
                else:
                    total_failed += 1
            elif success:
                total_processed += 1
            else:
                total_failed += 1
                
        except Exception as e:
            print(f"    ERROR processing {cpg_id}: {e}")
            total_failed += 1
            # Log error
            error_log = os.path.join(job_log_dir, "errors.log")
            with open(error_log, "a") as f:
                f.write(f"{cpg_id}: {e}\n")
        
        cpgs_since_restart += 1
    
    # Final cleanup
    stop_hail(tmp_dir)
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETED")
    print(f"{'='*60}")
    print(f"  Total CpGs: {len(cpg_ids)}")
    print(f"  Successfully processed: {total_processed}")
    print(f"  Failed: {total_failed}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    if len(cpg_ids) > 0:
        print(f"  Average time per CpG: {total_time/len(cpg_ids):.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

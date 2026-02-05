#!/usr/bin/env python
"""
Optimized Hail fine-mapping pipeline with genomic batching and checkpointing.

Key optimizations:
1. Genomic window batching - process 50 CpGs per batch with shared BlockMatrix filtering
2. Checkpointing - skip already completed CpGs by checking output files
3. Aggressive memory management - explicit caching/unpersisting
4. Fallback processing - retry failed CpGs individually
5. Enhanced Spark configuration for 65 CPUs / 250GB RAM
"""

import hail as hl
from hail.linalg import BlockMatrix
import os
import sys
import subprocess
import shutil
import pandas as pd
import numpy as np
import argparse
import time
import logging
import gc
from datetime import datetime
from typing import List, Dict, Set, Tuple
from collections import defaultdict


def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger('finemap')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def get_completed_cpgs(susie_out_dir: str, logger: logging.Logger) -> Set[str]:
    """Scan output directory for already completed CpGs."""
    completed = set()
    if not os.path.exists(susie_out_dir):
        return completed
    
    for f in os.listdir(susie_out_dir):
        if f.endswith('_susie.csv'):
            cpg_id = f.replace('_susie.csv', '')
            completed.add(cpg_id)
    
    logger.info(f"Found {len(completed)} already completed CpGs in {susie_out_dir}")
    return completed


def save_checkpoint(cpg_id: str, status: str, log_file: str):
    """Log completion status to checkpoint file."""
    timestamp = datetime.now().isoformat()
    with open(log_file, 'a') as f:
        f.write(f"{timestamp}\t{cpg_id}\t{status}\n")


def load_cpg_data(qtl_path: str, cpg_ids: List[str], logger: logging.Logger) -> pd.DataFrame:
    """Load only necessary columns and filter for relevant CpGs."""
    logger.info(f"Loading QTL data from {qtl_path}...")
    start = time.time()
    
    # Load data
    data = pd.read_csv(qtl_path)
    
    # Filter to relevant CpGs
    data = data[data['cpg'].isin(cpg_ids)]
    
    elapsed = time.time() - start
    logger.info(f"Loaded {len(data)} rows for {len(cpg_ids)} CpGs in {elapsed:.2f}s")
    
    return data


def create_cpg_batches(
    cpg_ids: List[str], 
    data: pd.DataFrame, 
    batch_size: int = 50,
    logger = None
) -> List[List[str]]:
    """
    Create batches of CpGs grouped by genomic proximity.
    
    Strategy:
    1. For each CpG, find lead SNP and define 3Mb window
    2. Sort by chromosome and position
    3. Create bins of ~50 CpGs that are genomically close
    4. Ensure each CpG has complete 3Mb window coverage
    """
    if logger:
        logger.info(f"Creating batches of size {batch_size}...")
    
    # Get lead SNP info for each CpG
    cpg_info = []
    for cpg_id in cpg_ids:
        subset = data[data['cpg'] == cpg_id]
        if len(subset) == 0:
            continue
        
        lead_snp = subset.loc[subset['pval'].idxmin()]
        cpg_info.append({
            'cpg_id': cpg_id,
            'chr': lead_snp['chr'],
            'pos': lead_snp['pos']
        })
    
    # Sort by chromosome then position
    cpg_info.sort(key=lambda x: (x['chr'], x['pos']))
    
    # Create batches
    batches = []
    current_batch = []
    
    for info in cpg_info:
        current_batch.append(info['cpg_id'])
        
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    
    # Add remaining CpGs
    if current_batch:
        batches.append(current_batch)
    
    if logger:
        logger.info(f"Created {len(batches)} batches from {len(cpg_info)} CpGs")
    
    return batches


def get_batch_snp_indices(
    batch_cpgs: List[str],
    data: pd.DataFrame,
    ht_idx: hl.Table,
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Get all SNP indices needed for all CpGs in a batch.
    
    Returns:
        - DataFrame with all SNPs in batch
        - Dict mapping cpg_id -> list of indices for that CpG
    """
    # Collect all SNPs for all CpGs in batch
    all_snp_data = []
    cpg_snp_map = {}
    
    for cpg_id in batch_cpgs:
        subset = data[data['cpg'] == cpg_id]
        
        if len(subset) == 0:
            cpg_snp_map[cpg_id] = []
            continue
        
        lead_snp = subset.loc[subset['pval'].idxmin()]
        snp_loc = lead_snp['pos']
        window_start = snp_loc - 1_500_000
        window_end = snp_loc + 1_500_000
        
        snp_df = subset[
            (subset['pos'].between(window_start, window_end)) &
            (subset['chr'] == lead_snp['chr'])
        ].copy()
        
        snp_df['cpg_id'] = cpg_id
        all_snp_data.append(snp_df)
        
        # Store count for validation
        cpg_snp_map[cpg_id] = len(snp_df)
    
    if not all_snp_data:
        return pd.DataFrame(), cpg_snp_map
    
    # Combine all SNPs
    batch_df = pd.concat(all_snp_data, ignore_index=True)
    batch_df['chr'] = batch_df['chr'].astype(str)
    
    return batch_df, cpg_snp_map


def match_variants(ht_snp: hl.Table, ht_idx: hl.Table) -> hl.Table:
    """Match variants with reference panel, handling strand flips."""
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
    
    return ht_matched


def process_single_cpg(
    cpg_id: str,
    data: pd.DataFrame,
    bm: BlockMatrix,
    ht_idx: hl.Table,
    out_dir: str,
    susie_out_dir: str,
    cleanup: bool,
    logger: logging.Logger
) -> bool:
    """Process a single CpG site."""
    start = time.time()
    logger.info(f"Processing CpG {cpg_id}...")
    
    subset = data[data['cpg'] == cpg_id]
    
    if len(subset) == 0:
        logger.warning(f"No data found for CpG {cpg_id}, skipping.")
        return False
    
    # Get lead SNP
    lead_snp = subset.loc[subset['pval'].idxmin()]
    snp_loc = lead_snp['pos']
    
    # Extract SNPs in 3Mb window
    window_start = snp_loc - 1_500_000
    window_end = snp_loc + 1_500_000
    snp_df = subset[
        (subset['pos'].between(window_start, window_end)) &
        (subset['chr'] == lead_snp['chr'])
    ].copy()
    
    snp_df['chr'] = snp_df['chr'].astype(str)
    
    if len(snp_df) == 0:
        logger.warning(f"No SNPs in window for CpG {cpg_id}, skipping.")
        return False
    
    # Convert to Hail format
    ht_snp = hl.Table.from_pandas(snp_df)
    ht_snp = ht_snp.annotate(
        locus=hl.locus(ht_snp.chr, ht_snp.pos),
        alleles=hl.array([ht_snp.allele2, ht_snp.allele1]),
    )
    ht_snp = ht_snp.key_by(locus=ht_snp.locus, alleles=ht_snp.alleles)
    ht_snp = ht_snp.cache()
    
    # Match variants
    ht_matched = match_variants(ht_snp, ht_idx)
    ht_matched = ht_matched.cache()
    
    # Order and convert to pandas
    ht_matched = ht_matched.order_by(ht_matched.idx)
    final_snp_df = ht_matched.to_pandas()
    
    if len(final_snp_df) == 0:
        logger.warning(f"No matched variants for CpG {cpg_id}, skipping.")
        ht_snp.unpersist()
        ht_matched.unpersist()
        return False
    
    # Log match counts
    n_forward = (~final_snp_df['flipped']).sum()
    n_flipped = (final_snp_df['flipped']).sum()
    logger.info(f"  Matched {len(final_snp_df)} variants ({n_forward} forward, {n_flipped} flipped)")
    
    # Flip signs and save
    final_snp_df['Z'] = np.where(final_snp_df['flipped'], -final_snp_df['Z'], final_snp_df['Z'])
    final_snp_df['AF'] = np.where(final_snp_df['flipped'], 1 - final_snp_df['AF'], final_snp_df['AF'])
    
    csv_file = f"{out_dir}/{cpg_id}.csv"
    final_snp_df.to_csv(csv_file, index=False)
    
    # Get indices and filter BlockMatrix
    idx = final_snp_df['idx'].tolist()
    bm_filtered = bm.filter(idx, idx)
    
    # Extract and symmetrize LD matrix
    ld_np = bm_filtered.to_numpy()
    ld_np = (ld_np + ld_np.T) / 2
    
    ld_file = f"{out_dir}/{cpg_id}_LD.txt"
    np.savetxt(ld_file, ld_np, delimiter=",")
    
    # Cleanup intermediate tables
    ht_snp.unpersist()
    ht_matched.unpersist()
    
    logger.info(f"  Data prep completed in {time.time() - start:.2f}s")
    
    return True


def process_batch_optimized(
    batch_cpgs: List[str],
    data: pd.DataFrame,
    bm: BlockMatrix,
    ht_idx: hl.Table,
    out_dir: str,
    susie_out_dir: str,
    cleanup: bool,
    completed_cpgs: Set[str],
    logger: logging.Logger
) -> Tuple[List[str], List[str]]:
    """
    Process a batch of CpGs with optimized BlockMatrix filtering.
    
    Returns:
        - List of successfully processed CpGs
        - List of failed CpGs
    """
    batch_start = time.time()
    logger.info(f"Processing batch of {len(batch_cpgs)} CpGs...")
    
    # Filter out already completed CpGs
    cpgs_to_process = [cpg for cpg in batch_cpgs if cpg not in completed_cpgs]
    skipped = len(batch_cpgs) - len(cpgs_to_process)
    if skipped > 0:
        logger.info(f"  Skipping {skipped} already completed CpGs")
    
    if not cpgs_to_process:
        logger.info(f"  All CpGs in batch already completed")
        return [], []
    
    # Get SNP indices for all CpGs in batch
    batch_df, cpg_counts = get_batch_snp_indices(cpgs_to_process, data, ht_idx, logger)
    
    if len(batch_df) == 0:
        logger.warning(f"  No SNPs found for any CpG in batch")
        return [], cpgs_to_process
    
    logger.info(f"  Total SNPs in batch: {len(batch_df)}")
    
    # Convert to Hail and match all at once
    ht_snp = hl.Table.from_pandas(batch_df)
    ht_snp = ht_snp.annotate(
        locus=hl.locus(ht_snp.chr, ht_snp.pos),
        alleles=hl.array([ht_snp.allele2, ht_snp.allele1]),
    )
    ht_snp = ht_snp.key_by(locus=ht_snp.locus, alleles=ht_snp.alleles)
    ht_snp = ht_snp.cache()
    
    ht_matched = match_variants(ht_snp, ht_idx)
    ht_matched = ht_matched.cache()
    
    # Get all unique indices for batch
    all_indices = ht_matched.aggregate(hl.agg.collect_as_set(ht_matched.idx))
    all_indices = sorted(list(all_indices))
    
    logger.info(f"  Unique SNP indices in batch: {len(all_indices)}")
    
    # Single BlockMatrix filter for entire batch
    if len(all_indices) > 0:
        logger.info(f"  Filtering BlockMatrix...")
        filter_start = time.time()
        bm_batch = bm.filter(all_indices, all_indices)
        logger.info(f"  BlockMatrix filter completed in {time.time() - filter_start:.2f}s")
    else:
        logger.warning(f"  No valid SNP indices found for batch")
        ht_snp.unpersist()
        ht_matched.unpersist()
        return [], cpgs_to_process
    
    # Convert matched table to pandas and split by CpG
    ht_matched = ht_matched.order_by(ht_matched.idx)
    matched_df = ht_matched.to_pandas()
    
    # OPTIMIZATION: Extract entire batch matrix to numpy ONCE
    # Then use numpy slicing for each CpG (much faster than BlockMatrix.filter)
    logger.info(f"  Extracting batch matrix to numpy...")
    extract_start = time.time()
    batch_matrix = bm_batch.to_numpy()
    # Symmetrize once for the whole batch matrix
    batch_matrix = (batch_matrix + batch_matrix.T) / 2
    logger.info(f"  Batch matrix extracted in {time.time() - extract_start:.2f}s "
                f"(shape: {batch_matrix.shape}, memory: {batch_matrix.nbytes / 1e6:.1f} MB)")
    
    # Create index mapping for submatrix extraction
    idx_to_position = {idx: pos for pos, idx in enumerate(all_indices)}
    
    # Process each CpG
    successful = []
    failed = []
    
    for cpg_id in cpgs_to_process:
        cpg_start = time.time()
        
        try:
            # Get SNPs for this CpG
            cpg_snps = matched_df[matched_df['cpg_id'] == cpg_id].copy()
            
            if len(cpg_snps) == 0:
                logger.warning(f"    No matched SNPs for {cpg_id}, skipping")
                failed.append(cpg_id)
                continue
            
            # Flip signs
            cpg_snps['Z'] = np.where(cpg_snps['flipped'], -cpg_snps['Z'], cpg_snps['Z'])
            cpg_snps['AF'] = np.where(cpg_snps['flipped'], 1 - cpg_snps['AF'], cpg_snps['AF'])
            
            # Save CSV
            csv_file = f"{out_dir}/{cpg_id}.csv"
            cpg_snps.to_csv(csv_file, index=False)
            
            # Get indices for this CpG
            cpg_indices = cpg_snps['idx'].tolist()
            positions = [idx_to_position[idx] for idx in cpg_indices]
            
            # OPTIMIZATION: Use numpy slicing instead of BlockMatrix.filter()
            # This is ~100-1000x faster than bm_batch.filter(positions, positions).to_numpy()
            ld_np = batch_matrix[np.ix_(positions, positions)]
            
            # Save LD matrix
            ld_file = f"{out_dir}/{cpg_id}_LD.txt"
            np.savetxt(ld_file, ld_np, delimiter=",")
            
            # Run SuSiE
            if run_susie(cpg_id, out_dir, susie_out_dir, cleanup, logger):
                successful.append(cpg_id)
                save_checkpoint(cpg_id, 'SUCCESS', 'completed.log')
            else:
                failed.append(cpg_id)
                save_checkpoint(cpg_id, 'FAILED: SuSiE', 'failed.log')
            
            logger.info(f"    {cpg_id} completed in {time.time() - cpg_start:.2f}s")
            
        except Exception as e:
            logger.error(f"    Error processing {cpg_id}: {e}")
            failed.append(cpg_id)
            save_checkpoint(cpg_id, f'FAILED: {e}', 'failed.log')
    
    # Cleanup
    ht_snp.unpersist()
    ht_matched.unpersist()
    bm_batch.unpersist()  # Unpersist the BlockMatrix
    del batch_matrix      # Free numpy matrix memory
    gc.collect()
    
    logger.info(f"  Batch completed in {time.time() - batch_start:.2f}s "
                f"({len(successful)} success, {len(failed)} failed)")
    
    return successful, failed


def process_single_cpg_fallback(
    cpg_id: str,
    data: pd.DataFrame,
    bm: BlockMatrix,
    ht_idx: hl.Table,
    out_dir: str,
    susie_out_dir: str,
    cleanup: bool,
    logger: logging.Logger
) -> bool:
    """Process a single CpG individually (fallback for batch failures)."""
    logger.info(f"  Processing {cpg_id} individually (fallback)...")
    
    try:
        if process_single_cpg(cpg_id, data, bm, ht_idx, out_dir, susie_out_dir, cleanup, logger):
            if run_susie(cpg_id, out_dir, susie_out_dir, cleanup, logger):
                save_checkpoint(cpg_id, 'SUCCESS', 'completed.log')
                return True
            else:
                save_checkpoint(cpg_id, 'FAILED: SuSiE', 'failed.log')
                return False
        else:
            save_checkpoint(cpg_id, 'FAILED: Data prep', 'failed.log')
            return False
    except Exception as e:
        logger.error(f"  Individual processing failed for {cpg_id}: {e}")
        save_checkpoint(cpg_id, f'FAILED: {e}', 'failed.log')
        return False


def run_susie(
    cpg_id: str,
    data_dir: str,
    out_dir: str,
    cleanup: bool,
    logger: logging.Logger
) -> bool:
    """Run SuSiE fine-mapping for a single CpG site."""
    start = time.time()
    
    csv_file = f"{data_dir}/{cpg_id}.csv"
    ld_file = f"{data_dir}/{cpg_id}_LD.txt"
    
    # Check input files exist
    if not os.path.exists(csv_file) or not os.path.exists(ld_file):
        logger.error(f"  Input files not found for {cpg_id}")
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
        
        logger.info(f"  SuSiE completed in {time.time() - start:.2f}s")
        
        # Clean up intermediate files if requested
        if cleanup:
            try:
                os.remove(csv_file)
                os.remove(ld_file)
                logger.info("  Cleaned up intermediate files")
            except Exception as e:
                logger.warning(f"  Could not clean up files: {e}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"  SuSiE ERROR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Optimized fine-mapping with genomic batching and checkpointing"
    )
    parser.add_argument(
        "--cpg-list",
        required=True,
        help="Comma-separated list of CpG IDs or path to file with one CpG per line",
    )
    parser.add_argument(
        "--qtl-path",
        default="../data/godmc/assoc_meta_for_finemapping.csv",
        help="Path to QTL data file"
    )
    parser.add_argument(
        "--output-dir",
        default="../data/finemapping_tmp/",
        help="Directory for intermediate files"
    )
    parser.add_argument(
        "--susie-out-dir",
        default="../data/susie_results/",
        help="Directory for SuSiE results"
    )
    parser.add_argument(
        "--log-dir",
        default="../logs/hail/",
        help="Directory for logs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of CpGs per batch (default: 25)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip already completed CpGs (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Process all CpGs even if already completed"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up intermediate files after successful processing"
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.susie_out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    if args.log_file is None:
        job_id = os.environ.get("PBS_ARRAY_INDEX", str(os.getpid()))
        args.log_file = os.path.join(args.log_dir, f"finemap_job_{job_id}.log")
    
    logger = setup_logging(args.log_file)
    logger.info("=" * 80)
    logger.info("Starting optimized fine-mapping pipeline")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"SuSiE output directory: {args.susie_out_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Resume mode: {args.resume}")
    logger.info("=" * 80)
    
    # Parse CpG list
    if os.path.exists(args.cpg_list):
        with open(args.cpg_list, "r") as f:
            cpg_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(cpg_ids)} CpG IDs from {args.cpg_list}")
    else:
        cpg_ids = [cpg.strip() for cpg in args.cpg_list.split(",")]
        logger.info(f"Processing {len(cpg_ids)} CpG IDs")
    
    # Check for already completed CpGs
    completed_cpgs = set()
    if args.resume:
        completed_cpgs = get_completed_cpgs(args.susie_out_dir, logger)
        remaining = [cpg for cpg in cpg_ids if cpg not in completed_cpgs]
        logger.info(f"Remaining CpGs to process: {len(remaining)}/{len(cpg_ids)}")
        cpg_ids = remaining
    
    if len(cpg_ids) == 0:
        logger.info("All CpGs already completed. Nothing to do.")
        return
    
    # Load QTL data for remaining CpGs
    data = load_cpg_data(args.qtl_path, cpg_ids, logger)
    
    if len(data) == 0:
        logger.error("No QTL data found for specified CpGs. Exiting.")
        return
    
    # Setup temp directory
    job_id = os.environ.get("PBS_ARRAY_INDEX", str(os.getpid()))
    ephemeral_base = os.environ.get("EPHEMERAL", os.path.dirname(os.path.abspath(args.output_dir)))
    tmp_dir = os.path.join(ephemeral_base, f"hail_tmp_{job_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    logger.info(f"Using temp directory: {tmp_dir}")
    
    # Spark configuration optimized for 65 CPUs / 250GB RAM
    spark_conf = {
        # S3 JARs
        "spark.jars": "/rds/general/user/tc1125/home/jars/hadoop-aws-3.3.4.jar,/rds/general/user/tc1125/home/jars/aws-java-sdk-bundle-1.12.539.jar",
        
        # Memory settings (200GB for Spark, leaving 50GB for system/R)
        "spark.driver.memory": "180g",
        "spark.executor.memory": "180g",
        "spark.memory.fraction": "0.85",
        "spark.memory.storageFraction": "0.6",
        "spark.driver.maxResultSize": "32g",
        
        # Broadcast optimization - critical for performance
        "spark.broadcast.compress": "true",
        "spark.broadcast.blockSize": "16m",
        "spark.broadcast.timeout": "1800s",
        "spark.broadcast.checksum": "true",
        
        # Serialization - Kryo is faster than Java serialization
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max": "2g",
        "spark.kryo.unsafe": "true",
        "spark.kryo.registrator": "is.hail.kryo.HailKryoRegistrator",
        
        # Performance tuning
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.default.parallelism": "130",  # 2x CPUs for I/O overlap
        "spark.sql.shuffle.partitions": "130",
        
        # GC tuning for large heap
        "spark.executor.extraJavaOptions": (
            "-XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+UseStringDeduplication"
        ),
        "spark.driver.extraJavaOptions": (
            "-XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+UseStringDeduplication"
        ),
        
        # S3 connection settings
        "spark.hadoop.fs.s3a.connection.maximum": "100",
        "spark.hadoop.fs.s3a.connection.timeout": "600000",
        "spark.hadoop.fs.s3a.connection.establish.timeout": "600000",
        "spark.hadoop.fs.s3a.attempts.maximum": "20",
        "spark.hadoop.fs.s3a.retry.limit": "20",
        "spark.hadoop.fs.s3a.retry.interval": "500ms",
        "spark.hadoop.fs.s3a.readahead.range": "256K",
        "spark.hadoop.fs.s3a.input.fadvise": "random",
        "spark.hadoop.fs.s3a.threads.max": "64",
        
        # Network and timeouts
        "spark.network.timeout": "600s",
        "spark.driver.bindAddress": "127.0.0.1",
        "spark.driver.host": "localhost",
        "spark.local.dir": tmp_dir,
        
        # Disable UI to avoid port conflicts
        "spark.ui.enabled": "false",
        "spark.driver.port": "0",
        "spark.blockManager.port": "0",
    }
    
    # Initialize Hail
    logger.info("Initializing Hail...")
    init_start = time.time()
    hl.init(
        master="local[*]",
        tmp_dir=tmp_dir,
        spark_conf=spark_conf,
    )
    logger.info(f"Hail initialized in {time.time() - init_start:.2f}s")
    
    # S3 paths for LD reference data
    ld_matrix_path = "s3a://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.bm"
    ld_variant_index_path = "s3a://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.variant.ht"
    
    # Load LD reference data
    logger.info("Loading LD reference data from S3...")
    ld_start = time.time()
    bm = BlockMatrix.read(ld_matrix_path)
    ht_idx = hl.read_table(ld_variant_index_path)
    ht_idx = ht_idx.cache()  # Cache the index table
    logger.info(f"LD reference data loaded in {time.time() - ld_start:.2f}s")
    
    # Create batches
    batches = create_cpg_batches(cpg_ids, data, args.batch_size, logger)
    
    # Process batches
    total_successful = 0
    total_failed = []
    batch_failures = []
    
    logger.info(f"\nProcessing {len(batches)} batches...\n")
    
    for i, batch in enumerate(batches, 1):
        logger.info(f"[{i}/{len(batches)}] Processing batch...")
        
        try:
            successful, failed = process_batch_optimized(
                batch, data, bm, ht_idx, args.output_dir, args.susie_out_dir,
                args.cleanup, completed_cpgs, logger
            )
            
            total_successful += len(successful)
            total_failed.extend(failed)
            
            # If batch had failures, try individually
            if failed:
                logger.info(f"  Retrying {len(failed)} failed CpGs individually...")
                for cpg_id in failed:
                    if process_single_cpg_fallback(
                        cpg_id, data, bm, ht_idx, args.output_dir,
                        args.susie_out_dir, args.cleanup, logger
                    ):
                        total_successful += 1
                        if cpg_id in total_failed:
                            total_failed.remove(cpg_id)
                    else:
                        batch_failures.append(cpg_id)
                        
        except Exception as e:
            logger.error(f"  Batch processing failed: {e}")
            logger.info(f"  Retrying all {len(batch)} CpGs individually...")
            
            for cpg_id in batch:
                if cpg_id in completed_cpgs:
                    continue
                    
                if process_single_cpg_fallback(
                    cpg_id, data, bm, ht_idx, args.output_dir,
                    args.susie_out_dir, args.cleanup, logger
                ):
                    total_successful += 1
                else:
                    total_failed.append(cpg_id)
                    batch_failures.append(cpg_id)
        
        # Periodic memory cleanup and progress report
        if i % 10 == 0:
            gc.collect()
            logger.info(f"\nProgress: {i}/{len(batches)} batches "
                       f"({total_successful} successful, {len(total_failed)} failed)\n")
    
    # Final summary
    logger.info("=" * 80)
    logger.info("Processing complete!")
    logger.info(f"Total successful: {total_successful}")
    logger.info(f"Total failed: {len(total_failed)}")
    logger.info(f"Already completed (skipped): {len(completed_cpgs)}")
    
    if total_failed:
        logger.info(f"\nFailed CpGs: {', '.join(total_failed)}")
        logger.info(f"\nFailed CpGs have been logged to failed.log")
        logger.info(f"Intermediate files (CSV/LD.txt) have been retained for debugging")
    
    logger.info("=" * 80)
    
    # Cleanup Hail
    logger.info("Stopping Hail...")
    hl.stop()
    
    # Cleanup temp directory
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp directory: {tmp_dir}")


if __name__ == "__main__":
    main()

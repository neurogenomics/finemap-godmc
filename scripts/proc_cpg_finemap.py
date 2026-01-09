# Prepare Z scores and LD matrices for finemapping.

import hail as hl
from hail.linalg import BlockMatrix
import os
import pandas as pd
import numpy as np
import argparse
import multiprocessing

def main():
    parser = argparse.ArgumentParser(description="Process data for finemapping")
    parser.add_argument('--cpg', required=True, help="CpG site identifier")
    parser.add_argument('--log-dir', default="../logs/hail/", help="Directory for Hail logs")

    args = parser.parse_args()
    cpg_id = args.cpg

    os.environ["HAIL_LOG_DIR"] = args.log_dir

    # Avoid timeouts according to https://discuss.hail.is/t/timeout-waiting-for-connection-from-pool-loading-gvcf-from-s3/2194/3
    n_cores = multiprocessing.cpu_count()
    max_conn = int(1.2 * n_cores)

    hl.init(
        spark_conf={
            "spark.jars": "/home/tobyc/data/jars/hadoop-aws-3.3.4.jar,/home/tobyc/data/jars/aws-java-sdk-bundle-1.12.539.jar",
            "spark.hadoop.fs.s3a.connection.maximum": str(max_conn),
        }
    )
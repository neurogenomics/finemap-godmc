#!/bin/bash
#PBS -l select=1:ncpus=65:mem=250gb
#PBS -l walltime=48:00:00
#PBS -N run_finemap
#PBS -J 1-20

cd "$PBS_O_WORKDIR"

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate finemapping

# AWS credentials
export AWS_ACCESS_KEY_ID=$(grep -A2 '\[default\]' ~/.aws/credentials | grep aws_access_key_id | cut -d'=' -f2 | tr -d ' ')
export AWS_SECRET_ACCESS_KEY=$(grep -A2 '\[default\]' ~/.aws/credentials | grep aws_secret_access_key | cut -d'=' -f2 | tr -d ' ')
export AWS_DEFAULT_REGION="us-east-1"

# Use ephemeral storage for Spark temp files (more space than /tmp)
export EPHEMERAL="${EPHEMERAL:-${TMPDIR:-/tmp}}"


IDS="../data/godmc/cpg_ids.txt"

TOTAL=$(wc -l < $IDS)
NUM_JOBS=20

TASK_ID=$((PBS_ARRAY_INDEX - 1))
CHUNK_SIZE=$(( (TOTAL + NUM_JOBS - 1) / NUM_JOBS ))
START=$(( TASK_ID * CHUNK_SIZE + 1 ))
END=$(( START + CHUNK_SIZE - 1 ))
if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi

echo "Job ${PBS_ARRAY_INDEX}: Processing CpGs ${START}-${END} ($((${END} - ${START} + 1)) CpGs)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

# Create temp directory in ephemeral storage for this job's files
TEMP_DIR="${EPHEMERAL}/finemap_job_${PBS_ARRAY_INDEX}"
mkdir -p "${TEMP_DIR}"
echo "Using temp directory: ${TEMP_DIR}"

# Extract this task's CpGs
sed -n "${START},${END}p" $IDS > "${TEMP_DIR}/cpgs.txt"

# Run optimized script with all paths relative to repo root
python scripts/run_cpg_finemap_optimized.py \
    --cpg-list "${TEMP_DIR}/cpgs.txt" \
    --batch-size 25 \
    --resume \
    --cleanup \
    --log-file logs/hail/finemap_job_${PBS_ARRAY_INDEX}.log \
    --output-dir ../data/finemapping_tmp/ \
    --susie-out-dir ../data/susie_results/

EXIT_CODE=$?

echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"

# Clean up temporary directory
rm -rf "${TEMP_DIR}"

exit ${EXIT_CODE}

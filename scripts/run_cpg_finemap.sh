#!/bin/bash
#PBS -l select=1:ncpus=65:mem=250gb
#PBS -l walltime=48:00:00
#PBS -N run_finemap
#PBS -J 1-20

cd $PBS_O_WORKDIR

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
echo "Using ${EPHEMERAL} for temporary files"
echo "Start time: $(date)"

# Extract this task's CpGs
sed -n "${START},${END}p" $IDS > cpgs_job_${PBS_ARRAY_INDEX}.txt

# Run optimized script
python run_cpg_finemap_optimized.py \
    --cpg-list cpgs_job_${PBS_ARRAY_INDEX}.txt \
    --batch-size 50 \
    --resume \
    --cleanup \
    --log-file ../logs/hail/finemap_job_${PBS_ARRAY_INDEX}.log

EXIT_CODE=$?

echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"

# Clean up temporary file
rm -f cpgs_job_${PBS_ARRAY_INDEX}.txt

exit ${EXIT_CODE}


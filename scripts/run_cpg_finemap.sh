#!/bin/bash
#PBS -l select=1:ncpus=24:mem=80gb
#PBS -l walltime=24:00:00
#PBS -N run_finemap
#PBS -J 1-90

cd $PBS_O_WORKDIR

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate finemapping

export AWS_ACCESS_KEY_ID=$(grep -A2 '\[default\]' ~/.aws/credentials | grep aws_access_key_id | cut -d'=' -f2 | tr -d ' ')
export AWS_SECRET_ACCESS_KEY=$(grep -A2 '\[default\]' ~/.aws/credentials | grep aws_secret_access_key | cut -d'=' -f2 | tr -d ' ')
export AWS_DEFAULT_REGION="us-east-1"

IDS="../data/godmc/cpg_ids.txt"

TOTAL=$(wc -l < $IDS)
NUM_JOBS=90

TASK_ID=$((PBS_ARRAY_INDEX - 1))
CHUNK_SIZE=$(( (TOTAL + NUM_JOBS - 1) / NUM_JOBS ))
START=$(( TASK_ID * CHUNK_SIZE + 1 ))
END=$(( START + CHUNK_SIZE - 1 ))
if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi

# extract this task’s CpGs
sed -n "${START},${END}p" $IDS > cpgs_${PBS_ARRAY_INDEX}.txt

python run_cpg_finemap.py --cpg-list cpgs_${PBS_ARRAY_INDEX}.txt --cleanup --run-susie


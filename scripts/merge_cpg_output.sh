#!/bin/bash
#PBS -l select=1:ncpus=65:mem=250gb
#PBS -l walltime=48:00:00
#PBS -N merge_susie

cd "$PBS_O_WORKDIR"

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate finemapping

python merge_cpg_output.py \
     --input-dir ../data/susie_results \
     --output-path ../data/susie_results_merged.csv

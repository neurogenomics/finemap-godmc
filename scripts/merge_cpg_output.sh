#!/bin/bash
#PBS -l select=1:ncpus=65:mem=250gb
#PBS -l walltime=48:00:00
#PBS -N run_finemap_local_idx

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate finemapping

python merge_cpg_output.py
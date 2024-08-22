#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --output=benchmark.out
#SBATCH --error=benchmark.err
#SBATCH -p xifu
#SBATCH --nodes=1
#SBATCH --array=0-100%160
#SBATCH --cpus-per-task=1

export JAX_ENABLE_X64=True
source /home/sila/miniconda3/etc/profile.d/conda.sh
conda activate jaxspec
. /xifu/usr/src/heasoft/heasoft-6.31.1/x86_64-pc-linux-gnu-libc2.26/headas-init.sh
cd /home/dups/ensemble-convergence

python reduce_mcmc.py $SLURM_ARRAY_TASK_ID 20
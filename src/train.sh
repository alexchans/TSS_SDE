#!/bin/bash
#SBATCH -J tsspro_sde_train
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -N 1 
#SBATCH -c 8
#SBATCH -p batch #short for test (1 gpu and 2 hours) and batch (8 gpus and 2 days)
#SBATCH --mem=64GB
#SBATCH --gres=gpu:8 #1 for short and 8 for batch

# ===== System to run on: change this one line (ala / tri / tetra) =====
SYS=tetra

# Per-system training log. 
exec > ../logs/run_${SYS}.log 2>&1

module load conda
conda activate tss_sde

cd $SLURM_SUBMIT_DIR/..
torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE src/train.py --epochs 500 --batch_size 32 --data data/${SYS}.npy --atoms data/${SYS}_atoms.txt
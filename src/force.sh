#!/bin/bash
#SBATCH -J tsspro_sde_force
#SBATCH -o ../logs/force.log
#SBATCH -e ../logs/force.log
#SBATCH -N 1 
#SBATCH -c 4
#SBATCH -p short
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

module load conda
conda activate tss_sde

# ===== System to run on: change this one line (ala / tri / tetra) =====
SYS=ala
STEPS=5000

cd $SLURM_SUBMIT_DIR/..
python -u src/utils/force.py data/${SYS}_gt.xyz trajectories/${SYS}/${SYS}_trajectories_${STEPS}.xyz --atoms data/${SYS}_atoms.txt --max_frames 500

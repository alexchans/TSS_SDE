#!/bin/bash
#SBATCH -J tsspro_sde_convert
#SBATCH -o ../logs/convert.log
#SBATCH -e ../logs/convert.log
#SBATCH -N 1 
#SBATCH -c 4
#SBATCH -p short
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

module load conda
conda activate tss_sde

# ===== System to run on: change this one line (ala / tri / tetra) =====
SYS=ala

cd $SLURM_SUBMIT_DIR/..
python -u src/utils/convert_npy_to_xyz.py data/${SYS}.npy --atoms data/${SYS}_atoms.txt --output data/${SYS}_gt.xyz --max_frames 50000

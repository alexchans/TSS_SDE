#!/bin/bash
#SBATCH -J tsspro_sde_gen
#SBATCH -o ../logs/generate.log
#SBATCH -e ../logs/generate.log
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
DIFFUSION_SCALE=0.4
# true -> deterministic ODE rollout (g=0), written with a _no_diffusion suffix so it
# sits alongside the stochastic trajectory rather than overwriting it. Set the same
# flag in the eval scripts to evaluate it. DIFFUSION_SCALE is ignored when true.
NO_DIFFUSION=false

FLAGS="--diffusion_scale ${DIFFUSION_SCALE}"
if [ "$NO_DIFFUSION" = true ]; then FLAGS="--no-diffusion"; fi

cd $SLURM_SUBMIT_DIR/..
python -u src/generate.py --model model/${SYS}_sde.pt --seed data/${SYS}.npy --atoms data/${SYS}_atoms.txt --num_steps $STEPS $FLAGS

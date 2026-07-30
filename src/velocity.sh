#!/bin/bash
#SBATCH -J tsspro_sde_vel
#SBATCH -o ../logs/velocity.log
#SBATCH -e ../logs/velocity.log
#SBATCH -N 1 
#SBATCH -c 4
#SBATCH -p short
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

module load conda
conda activate tss_sde

# ===== System to run on: change this one line (ala / tri / tetra) =====
SYS=tetra
STEPS=5000
# true -> evaluate the deterministic ODE trajectory (generated with --no-diffusion)
# instead of the stochastic one. Requires generate.sh to have produced it with the
# same flag; plots get the same suffix, so they never overwrite the stochastic ones.
NO_DIFFUSION=false

SUFFIX=""
if [ "$NO_DIFFUSION" = true ]; then SUFFIX="_no_diffusion"; fi

TRAJ=trajectories/${SYS}/${SYS}_trajectories_${STEPS}${SUFFIX}.xyz
# Plot name keeps the suffix last (..._velocity_no_diffusion.png) rather than inheriting the
# trajectory's (..._no_diffusion_velocity.png), so plots sort by metric.
PLOT=trajectories/${SYS}/${SYS}_trajectories_${STEPS}_velocity${SUFFIX}.png

cd $SLURM_SUBMIT_DIR/..
python -u src/utils/velocity.py ${TRAJ} --gt data/${SYS}_gt.xyz --output ${PLOT}

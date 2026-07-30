#!/bin/bash
#SBATCH -J tsspro_energy
#SBATCH -o ../logs/energy.log
#SBATCH -e ../logs/energy.log
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
# Plot name keeps the suffix last (..._energy_no_diffusion.png) rather than inheriting the
# trajectory's (..._no_diffusion_energy.png), so plots sort by metric.
PLOT=trajectories/${SYS}/${SYS}_trajectories_${STEPS}_energy${SUFFIX}.png

cd $SLURM_SUBMIT_DIR/..
python src/utils/energy.py data/${SYS}_gt.xyz ${TRAJ} --atoms data/${SYS}_atoms.txt --max_frames 500 --output ${PLOT}

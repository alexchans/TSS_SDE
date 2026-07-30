#!/bin/bash
#SBATCH -J tsspro_sde_ae
#SBATCH -o ../logs/autoencoder.log
#SBATCH -e ../logs/autoencoder.log
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
# true -> evaluate the deterministic ODE trajectory (generated with --no-diffusion)
# instead of the stochastic one. Requires generate.sh to have produced it with the
# same flag; plots get the same suffix, so they never overwrite the stochastic ones.
NO_DIFFUSION=true

SUFFIX=""
if [ "$NO_DIFFUSION" = true ]; then SUFFIX="_no_diffusion"; fi

TRAJ=trajectories/${SYS}/${SYS}_trajectories_${STEPS}${SUFFIX}.npy
# Plot name keeps the suffix last (..._latent_no_diffusion.png) rather than inheriting the
# trajectory's (..._no_diffusion_latent.png), so plots sort by metric.
PLOT=trajectories/${SYS}/${SYS}_trajectories_${STEPS}_latent${SUFFIX}.png

cd $SLURM_SUBMIT_DIR/..
python -u src/utils/autoencoder.py ${TRAJ} --gt data/${SYS}.npy --max_frames 5000 --output ${PLOT}

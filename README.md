# TSS_SDE

Protein MD trajectory generation with neural SDEs. A learned stochastic differential equation
models per-atom dynamics from molecular dynamics simulations and rolls out novel trajectories.

- **Drift** — an E(n)-equivariant graph neural network (EGNN) over a dynamic KNN graph, conditioned
  on time via FiLM modulation and sinusoidal embeddings.
- **Diffusion** — a learned per-atom, per-axis coefficient predicted in log space and clamped to a
  fixed range, so it fits the residual scale the drift cannot predict.

Both are trained jointly in a single phase, combining a noise-free drift rollout (position,
velocity, bond-length, and clash losses) with an Euler–Maruyama transition negative log-likelihood
teacher-forced on consecutive ground-truth frames.

## Setup

```bash
conda create -n tss_sde python=3.10 -y
conda activate tss_sde
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The energy and force evaluations additionally require [OpenMM](https://openmm.org) and
[pdbfixer](https://github.com/openmm/pdbfixer) (Amber14 with GBn2 implicit solvent).

## Data

Coordinates are `data/<name>.npy` with shape `(num_frames, num_atoms, 3)`, paired with
`data/<name>_atoms.txt` — one atom name per line, in the same order as the coordinate axis.
**Row order defines atom identity**: the model's node embeddings, the topology segmentation, and
`.xyz` element inference all depend on it. No PDB or topology file is needed; covalent bonds are
reconstructed from the atom names alone.

`data/ala.npy` (alanine dipeptide) ships with the repository as a runnable example. Larger
trajectories are gitignored — supply your own `.npy` plus matching `_atoms.txt` to train on a
different system.

## Running

Every `src/*.sh` script is an `sbatch` wrapper that `cd`s to the repository root before invoking
Python, so all paths inside them are relative to the root rather than to `src/`.

```bash
cd src
sbatch train.sh        # train (8 GPUs, up to 2 days)
sbatch generate.sh     # generate trajectories (1 GPU)
sbatch convert.sh      # convert .npy to .xyz
sbatch energy.sh       # compare potential energy (GT vs generated)
sbatch force.sh        # compare forces (GT vs generated)
sbatch velocity.sh     # compare velocity distributions
sbatch rmsd.sh         # RMSD over time
sbatch rama.sh         # Ramachandran plots
sbatch autocorr.sh     # coordinate autocorrelation
sbatch autoencoder.sh  # latent-space comparison
```

Arguments are hardcoded in the `python ...` line at the bottom of each script — edit that line to
change the dataset, step count, or ablation. Each script also sets a `SYS` variable near the top
selecting which system to run on.

**Without SLURM**, copy the final `python ...` line out of the relevant `.sh` and run it from the
repository root.

### Key arguments

**Training** (`train.py`) — `--data`, `--atoms`, `--epochs` (500), `--batch_size` (32),
`--max_frames` (50000), `--k` (default: fully connected), `--nll_weight` (0.1)

**Generation** (`generate.py`) — `--model`, `--seed`, `--atoms`, `--num_steps`,
`--diffusion_scale`, `--divergence_factor`, `--k`, `--no-drift`, `--no-diffusion`

> The ablation flags `--no-drift` and `--no-diffusion` zero the drift or diffusion term and append
> `_no_drift` / `_no_diffusion` to output filenames, so ablations sit alongside the full trajectory
> instead of overwriting it. The evaluation scripts take a matching flag.

**Evaluation** — all take a positional generated `.xyz` (energy and force take ground-truth and
generated `.xyz` as two positionals) plus `--output`:

| Script | Notable arguments |
| --- | --- |
| `utils/energy.py` | `--atoms`, `--max_frames`, `--no_minimize` |
| `utils/force.py` | `--atoms`, `--max_frames`, `--minimize`, `--save_npy` |
| `utils/velocity.py` | `--gt`, `--dt` |
| `utils/rmsd.py` | `--gt`, `--no-align`, `--ref-frame` |
| `utils/rama.py` | `--gt`, `--atoms`, `--max_frames` |
| `utils/autocorr.py` | `--gt`, `--nlags`, `--dt`, `--no-center` |
| `utils/autoencoder.py` | `--gt`, `--latent_dim`, `--epochs`, `--batch_size`, `--lr`, `--max_frames`, `--no-align`, `--seed` |
| `utils/convert_npy_to_xyz.py` | `--atoms`, `--max_frames` |

## Outputs

Generated trajectories are written to
`trajectories/<name>/<name>_trajectories_<steps>[suffix].{npy,xyz}`, with evaluation plots saved
alongside. Model weights land in `model/<name>_sde.pt` (best) and `model/<name>_sde_final.pt`
(last); both store exponential-moving-average weights. The `model/`, `logs/`, and `trajectories/`
directories are created on first run and are not tracked here.

Keep the naming stem (`ala`, `tri`, `tetra`, …) consistent across the `.npy`, the `_atoms.txt`,
the model path, and the output directory.

## Project structure

```
TSS_SDE/
├── src/
│   ├── sde_model.py              # GeometricSDE: EGNN drift + KNN graph + log-space diffusion
│   ├── train.py                  # Multi-GPU training, drift rollout + transition NLL
│   ├── generate.py               # Rolling-window trajectory generation
│   ├── *.sh                      # SLURM batch wrappers for each entry point
│   └── utils/
│       ├── dataset.py            # Sliding-window dataset
│       ├── topology.py           # Bond reconstruction from atom names
│       ├── energy.py             # → sbatch energy.sh
│       ├── force.py              # → sbatch force.sh
│       ├── velocity.py           # → sbatch velocity.sh
│       ├── rmsd.py               # → sbatch rmsd.sh
│       ├── rama.py               # → sbatch rama.sh
│       ├── autocorr.py           # → sbatch autocorr.sh
│       ├── autoencoder.py        # → sbatch autoencoder.sh
│       ├── convert_npy_to_xyz.py # → sbatch convert.sh
│       ├── remove_hydrogens.py   # strip hydrogens from a dataset
│       └── view_data.py          # inspect a .npy
├── data/                         # Coordinates (.npy) + atom names (.txt)
└── requirements.txt
```

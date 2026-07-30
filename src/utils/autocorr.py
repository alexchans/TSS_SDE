"""Compute and plot the coordinate autocorrelation function (ACF) from .xyz trajectories.

For each atom and Cartesian axis the per-frame coordinate is treated as a time series;
its autocorrelation is computed with statsmodels (statsmodels.tsa.stattools.acf) and then
averaged over all atoms and axes to give a mean ACF as a function of lag. The per-frame
center of mass is removed first (unless --no-center) so the ACF reflects internal
conformational relaxation rather than overall translational diffusion.

Optionally overlays the generated trajectory against ground truth and reports the
decorrelation time (first lag where the mean ACF drops below 1/e).
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')


def read_xyz(xyz_path):
    """Read .xyz trajectory into (num_frames, num_atoms, 3) array."""
    frames = []

    with open(xyz_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        i += 1  # skip comment line
        i += 1

        coords = []
        for j in range(num_atoms):
            parts = lines[i].split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1

        frames.append(coords)

    return np.array(frames)


def compute_mean_acf(coords, nlags, center=True):
    """Mean coordinate autocorrelation over all atoms and axes.

    Args:
        coords: (T, N, 3) trajectory coordinates in Å
        nlags: number of lags to compute (result has length nlags+1, lag 0 = 1)
        center: if True, subtract the per-frame center of mass before computing ACF

    Returns:
        (mean_acf, std_acf) each of shape (nlags+1,); std is across atom/axis series
    """
    if center:
        coords = coords - coords.mean(axis=1, keepdims=True)

    T, N, _ = coords.shape
    series = coords.reshape(T, N * 3).T  # (N*3, T), one time series per row

    acfs = []
    for s in series:
        if np.std(s) < 1e-8:
            continue  # skip constant series (ACF undefined)
        acfs.append(acf(s, nlags=nlags, fft=True))

    acfs = np.array(acfs)  # (num_series, nlags+1)
    return acfs.mean(axis=0), acfs.std(axis=0)


def decorrelation_time(mean_acf):
    """First lag at which the mean ACF drops below 1/e, or None if it never does."""
    below = np.where(mean_acf < 1.0 / np.e)[0]
    return int(below[0]) if len(below) else None


def plot_acf(mean_gen, std_gen, output_path, title="Coordinate ACF",
             mean_gt=None, std_gt=None, dt=1.0):
    """Plot mean ACF vs lag and save to file. Optionally overlay ground truth."""
    fig, ax = plt.subplots(figsize=(10, 5))
    lags = np.arange(len(mean_gen)) * dt

    tau_gen = decorrelation_time(mean_gen)
    tau_gt = decorrelation_time(mean_gt) if mean_gt is not None else None

    if mean_gt is not None:
        lags_gt = np.arange(len(mean_gt)) * dt
        ax.fill_between(lags_gt, mean_gt - std_gt, mean_gt + std_gt, color='#2563eb', alpha=0.15)
        gt_label = f"Ground Truth (τ={tau_gt*dt:.1f})" if tau_gt is not None else "Ground Truth (τ>range)"
        ax.plot(lags_gt, mean_gt, color='#2563eb', linewidth=1.5, label=gt_label)

    ax.fill_between(lags, mean_gen - std_gen, mean_gen + std_gen, color='#dc2626', alpha=0.15)
    gen_label = f"Generated (τ={tau_gen*dt:.1f})" if tau_gen is not None else "Generated (τ>range)"
    ax.plot(lags, mean_gen, color='#dc2626', linewidth=1.5, label=gen_label)

    ax.axhline(0.0, color='black', linewidth=0.8, ls=':')
    ax.axhline(1.0 / np.e, color='gray', linewidth=0.8, ls='--', alpha=0.6, label='1/e')

    ax.set_xlabel(f"Lag ({'frames' if dt == 1.0 else 'ps'})", fontsize=12)
    ax.set_ylabel("Autocorrelation", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved ACF plot: {output_path}")
    print(f"  Lags: {len(mean_gen) - 1}")
    tau_str = f"{tau_gen} lags" if tau_gen is not None else "> range"
    print(f"  Generated decorrelation time (1/e): {tau_str}")
    if mean_gt is not None:
        tau_gt_str = f"{tau_gt} lags" if tau_gt is not None else "> range"
        print(f"  Ground truth decorrelation time (1/e): {tau_gt_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot coordinate ACF from .xyz trajectory.")
    parser.add_argument("xyz", help="Path to generated .xyz trajectory file")
    parser.add_argument("--gt", default=None, help="Path to ground truth .xyz trajectory file")
    parser.add_argument("--output", default=None, help="Output plot path (default: <input>_acf.png)")
    parser.add_argument("--nlags", type=int, default=200, help="Number of lags to compute (default: 200)")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step between frames in ps (default: 1.0)")
    parser.add_argument("--no-center", action="store_true", help="Do not remove per-frame center of mass")
    args = parser.parse_args()

    print(f"Loading generated trajectory: {args.xyz}")
    gen_coords = read_xyz(args.xyz)
    print(f"  {gen_coords.shape[0]} frames, {gen_coords.shape[1]} atoms")

    nlags = min(args.nlags, gen_coords.shape[0] - 1)
    center = not args.no_center

    mean_gen, std_gen = compute_mean_acf(gen_coords, nlags, center=center)

    mean_gt = std_gt = None
    if args.gt:
        print(f"Loading ground truth trajectory: {args.gt}")
        gt_coords = read_xyz(args.gt)
        gt_coords = gt_coords[:len(gen_coords)]  # match frame count
        print(f"  {gt_coords.shape[0]} frames (matched to generated)")
        gt_nlags = min(nlags, gt_coords.shape[0] - 1)
        mean_gt, std_gt = compute_mean_acf(gt_coords, gt_nlags, center=center)

    if args.output is None:
        base = os.path.splitext(args.xyz)[0]
        args.output = base + "_acf.png"

    stem = os.path.splitext(os.path.basename(args.xyz))[0]
    system_name = stem.split('_trajectories')[0]
    # Flag ablation rollouts in the title — the plot filename alone is easy to mix up
    # when comparing against the stochastic run side by side.
    if stem.endswith('_no_diffusion'):
        system_name += " (no diffusion, ODE only)"
    elif stem.endswith('_no_drift'):
        system_name += " (no drift, diffusion only)"
    title = f"Coordinate Autocorrelation — {system_name}"
    plot_acf(mean_gen, std_gen, args.output, title=title,
             mean_gt=mean_gt, std_gt=std_gt, dt=args.dt)

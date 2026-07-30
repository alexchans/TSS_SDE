"""Compute and plot velocity distributions from .xyz trajectory files.

Computes velocities via central difference and plots:
  1. Speed magnitude KDE (all atoms & frames)
  2. Per-component (vx, vy, vz) KDE distributions
  3. Speed violin plot summary

Optionally compares generated trajectory against ground truth.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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


def compute_velocities(coords, dt):
    """Central difference: v[i] = (R[i+1] - R[i-1]) / (2*dt).

    Args:
        coords: (T, N, 3) trajectory coordinates in Å
        dt: time step between frames in ps

    Returns:
        (T-2, N, 3) velocities in Å/ps
    """
    v = (coords[2:] - coords[:-2]) / (2.0 * dt)
    return v


def plot_velocity(vel_gen, output_path, title="Velocity Distribution", vel_gt=None, dt=1.0):
    """Plot velocity distributions and save to file.

    Produces a 2×2 figure: speed KDE, vx KDE, vy KDE, vz KDE.
    Optionally overlays ground truth.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title}\n(central difference, 2Δt = {2*dt:.1f} ps)', fontsize=14, fontweight='bold')

    ax_speed = axes[0, 0]
    ax_vx = axes[0, 1]
    ax_vy = axes[1, 0]
    ax_vz = axes[1, 1]
    component_axes = {'vx (Å/ps)': (ax_vx, 0), 'vy (Å/ps)': (ax_vy, 1), 'vz (Å/ps)': (ax_vz, 2)}

    datasets = {'Generated': (vel_gen, '#dc2626')}
    if vel_gt is not None:
        datasets = {'Ground Truth': (vel_gt, '#2563eb'), 'Generated': (vel_gen, '#dc2626')}

    stats_lines = []

    for name, (v, color) in datasets.items():
        speeds = np.linalg.norm(v, axis=-1).ravel()

        # Speed KDE
        kde = gaussian_kde(speeds, bw_method='silverman')
        x_s = np.linspace(0, np.percentile(speeds, 99.5), 500)
        ax_speed.plot(x_s, kde(x_s), color=color, lw=2, label=name)
        ax_speed.axvline(speeds.mean(), color=color, lw=1.2, ls='--', alpha=0.7)

        # Per-component KDE
        for label, (ax, dim) in component_axes.items():
            comp = v[:, :, dim].ravel()
            kde_c = gaussian_kde(comp, bw_method='silverman')
            x_c = np.linspace(np.percentile(comp, 0.5), np.percentile(comp, 99.5), 500)
            ax.plot(x_c, kde_c(x_c), color=color, lw=2, label=name)
            ax.axvline(comp.mean(), color=color, lw=1, ls='--', alpha=0.6)

        stats_lines.append(
            f"  {name:15s}  speed: μ={speeds.mean():.4f}  σ={speeds.std():.4f} Å/ps  |"
            f"  vx: μ={v[:,:,0].ravel().mean():+.4f}  σ={v[:,:,0].ravel().std():.4f}"
            f"  vy: μ={v[:,:,1].ravel().mean():+.4f}  σ={v[:,:,1].ravel().std():.4f}"
            f"  vz: μ={v[:,:,2].ravel().mean():+.4f}  σ={v[:,:,2].ravel().std():.4f}"
        )

    ax_speed.set_xlabel('Speed |v| (Å/ps)')
    ax_speed.set_ylabel('Density')
    ax_speed.set_title(f'Speed magnitude distribution (all atoms & frames) — {title}')
    ax_speed.legend(framealpha=0.8)
    ax_speed.grid(True, alpha=0.3)

    for label, (ax, dim) in component_axes.items():
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_title(f'{label} distribution')
        ax.legend(framealpha=0.8, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', lw=0.8, ls=':')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved velocity distribution plot: {output_path}")
    print(f"  Velocity statistics:")
    for line in stats_lines:
        print(line)


def plot_velocity_violin(vel_gen, output_path, vel_gt=None, system_name=''):
    """Plot speed violin comparison and save to file."""
    fig, ax = plt.subplots(figsize=(8, 6))

    datasets = {}
    if vel_gt is not None:
        datasets['Ground Truth'] = vel_gt
    datasets['Generated'] = vel_gen

    colors = {'Ground Truth': '#2563eb', 'Generated': '#dc2626'}
    names_list = list(datasets.keys())
    speed_arrays = [np.linalg.norm(datasets[n], axis=-1).ravel() for n in names_list]

    # Downsample for violin (can be very large)
    rng = np.random.default_rng(42)
    speed_sampled = [rng.choice(s, size=min(len(s), 200000), replace=False) for s in speed_arrays]

    vp = ax.violinplot(speed_sampled, positions=range(len(names_list)),
                       showmeans=True, showmedians=True, widths=0.6)
    for i, (pc, n) in enumerate(zip(vp['bodies'], names_list)):
        pc.set_facecolor(colors[n])
        pc.set_alpha(0.75)

    for i, (n, arr) in enumerate(zip(names_list, speed_arrays)):
        ax.text(i, np.percentile(arr, 99) * 1.02,
                f'μ={arr.mean():.3f}\nσ={arr.std():.3f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(names_list)))
    ax.set_xticklabels(names_list)
    ax.set_ylabel('Speed |v| (Å/ps)')
    ax.set_title(f'Speed distribution per trajectory (violin) — {system_name}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved velocity violin plot: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot velocity distributions from .xyz trajectory.")
    parser.add_argument("xyz", help="Path to generated .xyz trajectory file")
    parser.add_argument("--gt", default=None, help="Path to ground truth .xyz trajectory file")
    parser.add_argument("--output", default=None, help="Output plot path (default: <input>_velocity.png)")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step between frames in ps (default: 1.0)")
    args = parser.parse_args()

    print(f"Loading generated trajectory: {args.xyz}")
    gen_coords = read_xyz(args.xyz)
    print(f"  {gen_coords.shape[0]} frames, {gen_coords.shape[1]} atoms")

    vel_gen = compute_velocities(gen_coords, args.dt)
    print(f"  {vel_gen.shape[0]} velocity frames (central difference, dt={args.dt} ps)")

    vel_gt = None
    if args.gt:
        print(f"Loading ground truth trajectory: {args.gt}")
        gt_coords = read_xyz(args.gt)
        gt_coords = gt_coords[:len(gen_coords)]  # match frame count
        print(f"  {gt_coords.shape[0]} frames (matched to generated)")
        vel_gt = compute_velocities(gt_coords, args.dt)
        print(f"  {vel_gt.shape[0]} velocity frames")

    if args.output is None:
        base = os.path.splitext(args.xyz)[0]
        args.output = base + "_velocity.png"

    system_name = os.path.splitext(os.path.basename(args.xyz))[0].split('_trajectories')[0]
    title = f"Velocity — {system_name}"
    plot_velocity(vel_gen, args.output, title=title, vel_gt=vel_gt, dt=args.dt)

    # Also save violin plot
    violin_path = os.path.splitext(args.output)[0] + "_violin.png"
    plot_velocity_violin(vel_gen, violin_path, vel_gt=vel_gt, system_name=system_name)

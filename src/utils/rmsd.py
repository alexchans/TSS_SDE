"""Compute and plot per-frame RMSD from a .xyz trajectory file."""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def read_xyz(xyz_path):
    """Read .xyz trajectory into (num_frames, num_atoms, 3) array."""
    frames = []
    elements = None

    with open(xyz_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        i += 1  # skip comment line
        comment = lines[i].strip()
        i += 1

        frame_elements = []
        coords = []
        for j in range(num_atoms):
            parts = lines[i].split()
            frame_elements.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1

        frames.append(coords)
        if elements is None:
            elements = frame_elements

    return np.array(frames), elements


def kabsch_rmsd(ref, target):
    """RMSD after Kabsch alignment (optimal superposition)."""
    # Center both structures
    ref_centered = ref - ref.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    # Kabsch: find optimal rotation via SVD
    H = target_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    diag = np.diag([1.0, 1.0, d])
    R = Vt.T @ diag @ U.T

    # Apply rotation and compute RMSD
    target_aligned = target_centered @ R.T
    diff = ref_centered - target_aligned
    return np.sqrt((diff ** 2).sum() / len(ref))


def simple_rmsd(ref, target):
    """Compute RMSD without alignment (just translation to center)."""
    ref_centered = ref - ref.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    diff = ref_centered - target_centered
    return np.sqrt((diff ** 2).sum() / len(ref))


def plot_rmsd(rmsd_values, output_path, title="RMSD vs Frame", gt_rmsd_values=None):
    """Plot RMSD over trajectory and save to file. Optionally overlay ground truth."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if gt_rmsd_values is not None:
        ax.plot(gt_rmsd_values, color='#2563eb', linewidth=0.8, alpha=0.7, label=f"Ground Truth (mean: {np.mean(gt_rmsd_values):.3f} Å)")
        ax.axhline(np.mean(gt_rmsd_values), color='#2563eb', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.plot(rmsd_values, color='#dc2626', linewidth=0.8, alpha=0.7, label=f"Generated (mean: {np.mean(rmsd_values):.3f} Å)")
        ax.axhline(np.mean(rmsd_values), color='#dc2626', linestyle='--', linewidth=0.8, alpha=0.4)
    else:
        ax.plot(rmsd_values, color='#2563eb', linewidth=1.0, alpha=0.9)
        ax.axhline(np.mean(rmsd_values), color='#dc2626', linestyle='--', linewidth=0.8, alpha=0.6, label=f"Mean: {np.mean(rmsd_values):.3f} Å")

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("RMSD (Å)", fontsize=12)
    if gt_rmsd_values is not None:
        ax.set_title(
            f"{title}\n"
            f"Gen — Max: {np.max(rmsd_values):.3f} Å  |  Mean: {np.mean(rmsd_values):.3f} Å\n"
            f"GT  — Max: {np.max(gt_rmsd_values):.3f} Å  |  Mean: {np.mean(gt_rmsd_values):.3f} Å",
            fontsize=12
        )
    else:
        ax.set_title(f"{title}\nMax: {np.max(rmsd_values):.3f} Å  |  Mean: {np.mean(rmsd_values):.3f} Å", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved RMSD plot: {output_path}")
    print(f"  Frames: {len(rmsd_values)}")
    print(f"  Mean RMSD: {np.mean(rmsd_values):.4f} Å")
    print(f"  Max RMSD:  {np.max(rmsd_values):.4f} Å")
    if gt_rmsd_values is not None:
        print(f"  GT Mean RMSD: {np.mean(gt_rmsd_values):.4f} Å")
        print(f"  GT Max RMSD:  {np.max(gt_rmsd_values):.4f} Å")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot RMSD from .xyz trajectory.")
    parser.add_argument("xyz", help="Path to generated .xyz trajectory file")
    parser.add_argument("--gt", default=None, help="Path to ground truth .xyz trajectory file")
    parser.add_argument("--output", default=None, help="Output plot path (default: same name as input .png)")
    parser.add_argument("--no-align", action="store_true", help="Skip Kabsch alignment (just center)")
    parser.add_argument("--ref-frame", type=int, default=0, help="Reference frame index (default: 0)")
    args = parser.parse_args()

    coords, elements = read_xyz(args.xyz)
    ref = coords[args.ref_frame]

    rmsd_fn = simple_rmsd if args.no_align else kabsch_rmsd
    rmsd_values = np.array([rmsd_fn(ref, coords[i]) for i in range(len(coords))])

    gt_rmsd_values = None
    if args.gt:
        gt_coords, _ = read_xyz(args.gt)
        gt_coords = gt_coords[:len(coords)]  # match number of generated frames
        gt_ref = gt_coords[0]
        gt_rmsd_values = np.array([rmsd_fn(gt_ref, gt_coords[i]) for i in range(len(gt_coords))])
        print(f"Ground truth: {len(gt_coords)} frames (matched to generated)")

    if args.output is None:
        base = os.path.splitext(args.xyz)[0]
        args.output = base + "_rmsd.png"

    stem = os.path.splitext(os.path.basename(args.xyz))[0]
    system_name = stem.split('_trajectories')[0]
    # Flag ablation rollouts in the title — the plot filename alone is easy to mix up
    # when comparing against the stochastic run side by side.
    if stem.endswith('_no_diffusion'):
        system_name += " (no diffusion, ODE only)"
    elif stem.endswith('_no_drift'):
        system_name += " (no drift, diffusion only)"
    title = f"RMSD vs Frame — {system_name}"
    plot_rmsd(rmsd_values, args.output, title=title, gt_rmsd_values=gt_rmsd_values)

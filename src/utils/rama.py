"""Compare Ramachandran (φ/ψ) distributions between ground truth and generated trajectories.

Computes backbone dihedral angles from .xyz trajectories using the atom names file
to locate N, CA, C atoms per residue. A chain of M residues has M-1 backbone
junctions, each with its own (φ, ψ) pair; every junction gets its own row of
side-by-side GT vs generated 2D density histograms (ala → 1 junction, tri → 2,
tetra → 3).

Works for all-atom systems (e.g. alanine dipeptide and longer peptides) where
N, CA, C are present on every residue.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def read_xyz(xyz_path):
    """Read .xyz trajectory into (num_frames, num_atoms, 3) array."""
    frames = []
    with open(xyz_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        i += 2  # skip count + comment
        coords = []
        for _ in range(num_atoms):
            parts = lines[i].split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1
        frames.append(np.array(coords))
    return np.array(frames)


def parse_residues(atom_names_path):
    """Parse atom names file and segment into residues (each starts with backbone N).

    Returns list of dicts: [{'N': global_idx, 'CA': global_idx, 'C': global_idx}, ...]
    """
    with open(atom_names_path) as f:
        all_atoms = [line.strip() for line in f if line.strip()]

    residues = []
    current = {}
    for i, name in enumerate(all_atoms):
        if name == 'N':
            if current:
                residues.append(current)
            current = {}
        if name in ('N', 'CA', 'C'):
            current[name] = i
    if current:
        residues.append(current)

    return residues


def dihedral_angle(p0, p1, p2, p3):
    """Compute dihedral angle (in degrees) for arrays of 4-point sets.

    Each argument: (num_frames, 3)
    Returns: (num_frames,) angles in [-180, 180].
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize
    n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True)
    n2_norm = np.linalg.norm(n2, axis=-1, keepdims=True)
    n1 = n1 / np.clip(n1_norm, 1e-10, None)
    n2 = n2 / np.clip(n2_norm, 1e-10, None)

    # Unit vector along b2
    b2_hat = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-10, None)

    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(np.cross(n1, n2) * b2_hat, axis=-1)

    return np.degrees(np.arctan2(y, x))


def compute_dihedrals(coords, residues):
    """Compute the (φ, ψ) angle pair at every backbone junction across all frames.

    A chain of M residues has M-1 junctions. Junction k (joining residue k to
    residue k+1) pairs:
        ψ = dihedral(N[k],  CA[k],  C[k],  N[k+1])   (ψ of residue k)
        φ = dihedral(C[k],  N[k+1], CA[k+1], C[k+1]) (φ of residue k+1)
    i.e. the two dihedrals on either side of the peptide bond between the two
    residues. Keeping each junction separate (rather than pooling all φ and all
    ψ) is what keeps φ paired with the correct ψ for multi-residue peptides.

    Args:
        coords: (num_frames, num_atoms, 3)
        residues: list of dicts from parse_residues()

    Returns:
        junctions: list of dicts ordered by junction, one per junction with a
            complete backbone:
            [{'k': k, 'phi': (num_frames,), 'psi': (num_frames,)}, ...]
    """
    junctions = []

    for k in range(len(residues) - 1):
        r0, r1 = residues[k], residues[k + 1]
        if not all(a in r0 for a in ('N', 'CA', 'C')):
            continue
        if not all(a in r1 for a in ('N', 'CA', 'C')):
            continue

        psi = dihedral_angle(
            coords[:, r0['N']],
            coords[:, r0['CA']],
            coords[:, r0['C']],
            coords[:, r1['N']],
        )
        phi = dihedral_angle(
            coords[:, r0['C']],
            coords[:, r1['N']],
            coords[:, r1['CA']],
            coords[:, r1['C']],
        )
        junctions.append({'k': k, 'phi': phi, 'psi': psi})

    return junctions


def plot_ramachandran(gen_junctions, output_path, gt_junctions=None, system_name=''):
    """Plot a Ramachandran comparison with one row per backbone junction.

    Each row is a junction: [Ground Truth | Generated] when GT is supplied,
    otherwise a single Generated panel. GT and generated junctions are matched
    by junction index `k` (same residue topology underlies both).
    """
    if not gen_junctions:
        print("No backbone junctions found — nothing to plot.")
        return

    has_gt = bool(gt_junctions)
    gt_by_k = {j['k']: j for j in gt_junctions} if has_gt else {}

    n_rows = len(gen_junctions)
    n_cols = 2 if has_gt else 1
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False
    )

    if system_name:
        fig.suptitle(f'Ramachandran Plot — {system_name}', fontsize=15, fontweight='bold')

    bins = 80
    hist_range = [[-180, 180], [-180, 180]]
    ticks = [-180, -120, -60, 0, 60, 120, 180]
    cmap = 'inferno'

    def style_ax(ax, title):
        ax.set_xlabel(r'$\phi$ (degrees)', fontsize=12)
        ax.set_ylabel(r'$\psi$ (degrees)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    def draw(ax, junction, label):
        _, _, _, im = ax.hist2d(
            junction['phi'], junction['psi'], bins=bins,
            range=hist_range, cmap=cmap, cmin=1,
        )
        style_ax(ax, f'{label} ({len(junction["phi"])} samples)')
        fig.colorbar(im, ax=ax, label='Density', shrink=0.8)

    for row, gen in enumerate(gen_junctions):
        k = gen['k']
        junction_label = f'Residues {k}–{k + 1}'
        col = 0
        if has_gt:
            gt = gt_by_k.get(k)
            if gt is not None:
                draw(axes[row][0], gt, f'Ground Truth — {junction_label}')
            else:
                axes[row][0].axis('off')
            col = 1
        draw(axes[row][col], gen, f'Generated — {junction_label}')

    plt.tight_layout(rect=(0, 0, 1, 0.98) if system_name else None)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Ramachandran plot: {output_path}")

    # Print per-junction summary statistics
    for row, gen in enumerate(gen_junctions):
        k = gen['k']
        print(f"  Junction {k}–{k + 1}:")
        pairs = [('Generated', gen)]
        if has_gt and k in gt_by_k:
            pairs.insert(0, ('Ground Truth', gt_by_k[k]))
        for name, j in pairs:
            print(f"    {name:15s}  φ: μ={j['phi'].mean():+7.1f}° σ={j['phi'].std():5.1f}°  |"
                  f"  ψ: μ={j['psi'].mean():+7.1f}° σ={j['psi'].std():5.1f}°")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Ramachandran distributions from .xyz trajectories.")
    parser.add_argument("xyz", help="Path to generated .xyz trajectory file")
    parser.add_argument("--gt", default=None, help="Path to ground truth .xyz trajectory file")
    parser.add_argument("--atoms", required=True, help="Path to atom names file")
    parser.add_argument("--output", default=None, help="Output plot path (default: <input>_rama.png)")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames to process")
    args = parser.parse_args()

    residues = parse_residues(args.atoms)
    print(f"Detected {len(residues)} residues from {args.atoms}")

    print(f"Loading generated trajectory: {args.xyz}")
    gen_coords = read_xyz(args.xyz)
    if args.max_frames:
        gen_coords = gen_coords[:args.max_frames]
    print(f"  {gen_coords.shape[0]} frames, {gen_coords.shape[1]} atoms")

    gen_junctions = compute_dihedrals(gen_coords, residues)
    print(f"  Computed {len(gen_junctions)} backbone junction(s)")

    gt_junctions = None
    if args.gt:
        print(f"Loading ground truth trajectory: {args.gt}")
        gt_coords = read_xyz(args.gt)
        gt_coords = gt_coords[:len(gen_coords)]  # match frame count
        if args.max_frames:
            gt_coords = gt_coords[:args.max_frames]
        print(f"  {gt_coords.shape[0]} frames (matched to generated), {gt_coords.shape[1]} atoms")

        gt_junctions = compute_dihedrals(gt_coords, residues)
        print(f"  Computed {len(gt_junctions)} backbone junction(s)")

    if args.output is None:
        base = os.path.splitext(args.xyz)[0]
        args.output = base + "_rama.png"

    stem = os.path.splitext(os.path.basename(args.xyz))[0]
    system_name = stem.split('_trajectories')[0]
    # Flag ablation rollouts in the title — the plot filename alone is easy to mix up
    # when comparing against the stochastic run side by side.
    if stem.endswith('_no_diffusion'):
        system_name += " (no diffusion, ODE only)"
    elif stem.endswith('_no_drift'):
        system_name += " (no drift, diffusion only)"
    plot_ramachandran(gen_junctions, args.output, gt_junctions=gt_junctions, system_name=system_name)

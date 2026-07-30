"""Convert a .npy trajectory file to .xyz format for 3D visualization."""

import argparse
import os
import numpy as np


def npy_to_xyz(npy_path, atom_names_path, output_path=None, max_frames=None):
    """Convert .npy coordinates to .xyz file."""
    trajectory = np.load(npy_path)
    if trajectory.ndim == 2:
        trajectory = trajectory.reshape(trajectory.shape[0], -1, 3)

    num_frames, num_atoms, _ = trajectory.shape
    if max_frames:
        num_frames = min(num_frames, max_frames)

    with open(atom_names_path) as f:
        atom_names = [line.strip() for line in f if line.strip()]

    elements = []
    for name in atom_names:
        if name.startswith('H'):
            elements.append('H')
        elif name in ('CA', 'CB'):
            elements.append('C')
        else:
            elements.append(name[0])

    if output_path is None:
        base = os.path.splitext(npy_path)[0]
        output_path = base + '.xyz'

    with open(output_path, 'w') as out:
        for frame_idx in range(num_frames):
            out.write(f"{num_atoms}\n")
            out.write(f"Frame {frame_idx}\n")
            for atom_idx in range(num_atoms):
                x, y, z = trajectory[frame_idx, atom_idx]
                out.write(f"{elements[atom_idx]:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")

    print(f"Converted {num_frames} frames ({num_atoms} atoms each)")
    print(f"  Input:  {npy_path}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy trajectory to .xyz format.")
    parser.add_argument("npy", help="Path to .npy trajectory file")
    parser.add_argument("--atoms", default="data/ala_atoms.txt", help="Path to atom names file")
    parser.add_argument("--output", default=None, help="Output .xyz path (default: same name as input)")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames to convert")
    args = parser.parse_args()

    npy_to_xyz(args.npy, args.atoms, args.output, args.max_frames)

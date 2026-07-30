"""Remove hydrogen atoms from coordinate and atom name files."""

import argparse
import numpy as np


def remove_hydrogens(coords_path, atom_names_path, out_coords, out_atoms):
    # Load atom names
    with open(atom_names_path) as f:
        atom_names = [line.strip() for line in f if line.strip()]

    # Identify non-hydrogen indices
    heavy_indices = [i for i, name in enumerate(atom_names) if not name.startswith('H')]
    heavy_names = [atom_names[i] for i in heavy_indices]

    print(f"Total atoms:        {len(atom_names)}")
    print(f"Hydrogen atoms:     {len(atom_names) - len(heavy_indices)}")
    print(f"Heavy atoms kept:   {len(heavy_indices)}")
    print(f"Heavy atom names:   {heavy_names}")
    print(f"Heavy atom indices: {heavy_indices}")

    # Load and filter coordinates
    coords = np.load(coords_path)
    print(f"\nOriginal shape: {coords.shape}")

    coords_noH = coords[:, heavy_indices, :]
    print(f"Filtered shape: {coords_noH.shape}")

    # Save
    np.save(out_coords, coords_noH)
    print(f"\nSaved coordinates: {out_coords}")

    with open(out_atoms, 'w') as f:
        for name in heavy_names:
            f.write(name + '\n')
    print(f"Saved atom names:  {out_atoms}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove hydrogen atoms from coordinate data.")
    parser.add_argument("--coords", default="data/ala.npy", help="Input coordinates .npy file")
    parser.add_argument("--atoms", default="data/ala_atoms.txt", help="Input atom names file")
    parser.add_argument("--out_coords", default="data/ala.npy", help="Output coordinates .npy file")
    parser.add_argument("--out_atoms", default="data/ala_atoms.txt", help="Output atom names file")
    args = parser.parse_args()

    remove_hydrogens(args.coords, args.atoms, args.out_coords, args.out_atoms)

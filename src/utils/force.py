"""Compare per-frame force magnitudes between ground truth and generated trajectories.

Uses OpenMM with Amber14 force field + implicit solvent (GBn2) for physically accurate forces.
Produces a comparison plot of force magnitude distributions and per-frame averages.
"""

import argparse
import io
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from openmm.app import ForceField, Simulation, NoCutoff
from openmm import VerletIntegrator
from openmm.unit import kilojoule_per_mole, nanometer
from pdbfixer import PDBFixer


def read_xyz(xyz_path):
    """Read .xyz trajectory into list of (num_atoms, 3) arrays."""
    frames = []
    if not os.path.exists(xyz_path):
        print(f"Error: File not found {xyz_path}")
        return []
        
    with open(xyz_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        try:
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            num_atoms = int(line)
            i += 2  # skip count + comment
            coords = []
            for _ in range(num_atoms):
                parts = lines[i].split()
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                i += 1
            frames.append(np.array(coords))
        except (ValueError, IndexError):
            break
    return frames


def build_atom_map(atom_names_path):
    """Build atom mapping from atom names file."""
    # Add src/ to path so we can import topology
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from utils.topology import identify_residue_type

    with open(atom_names_path) as f:
        atom_names = [line.strip() for line in f if line.strip()]

    # Segment into residues (each starts with backbone 'N')
    residues = []
    current_atoms = []
    for name in atom_names:
        if name == 'N' and current_atoms:
            residues.append(current_atoms)
            current_atoms = []
        current_atoms.append(name)
    if current_atoms:
        residues.append(current_atoms)

    # Build atom map with correct residue types
    atom_map = []
    for resnum, res_atoms in enumerate(residues, start=1):
        resname = identify_residue_type(res_atoms)
        for name in res_atoms:
            atom_map.append((resname, resnum, name))

    return atom_map


def frame_to_pdb(coords, atom_map):
    """Convert one frame to PDB string. coords: (N, 3) in Angstrom."""
    lines = []
    for i, (resname, resnum, atomname) in enumerate(atom_map):
        elem = atomname[0]
        x, y, z = coords[i]
        line = (
            f"ATOM  {i+1:5d} {atomname:<4s} {resname} A{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}  "
        )
        lines.append(line)
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def compute_force_magnitudes(frames, atom_map, forcefield_files, minimize=False):
    """Compute per-atom force magnitudes for each frame using OpenMM."""
    ff = ForceField(*forcefield_files)
    all_magnitudes = []  # List of (num_atoms,) arrays
    frame_averages = []
    failed = []

    # Use a small unit conversion factor: 1.0 kJ/mol/nm
    force_unit = kilojoule_per_mole / nanometer

    for i, coords in enumerate(tqdm(frames, desc="Computing forces", ncols=80)):
        try:
            pdb_str = frame_to_pdb(coords, atom_map)
            fixer = PDBFixer(pdbfile=io.StringIO(pdb_str))
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(pH=7.0)

            system = ff.createSystem(fixer.topology, nonbondedMethod=NoCutoff, constraints=None)
            integrator = VerletIntegrator(0.001)
            sim = Simulation(fixer.topology, system, integrator)
            sim.context.setPositions(fixer.positions)

            if minimize:
                sim.minimizeEnergy(maxIterations=20)

            state = sim.context.getState(getForces=True)
            forces = state.getForces(asNumpy=True).value_in_unit(force_unit)
            
            # Calculate magnitudes for each atom in this frame
            mags = np.linalg.norm(forces, axis=-1)  # (N_atoms,)
            all_magnitudes.append(mags)
            frame_averages.append(np.mean(mags))
            
        except Exception as e:
            frame_averages.append(np.nan)
            failed.append((i, str(e)))

    if failed:
        print(f"  Warning: {len(failed)} frames failed")

    return np.array(frame_averages), all_magnitudes


def plot_force_comparison(gt_avg, gt_all, gen_avg, gen_all, output_path, system_name=''):
    """Plot comparison of force magnitudes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Per-frame average force magnitude
    n = min(len(gt_avg), len(gen_avg))
    ax1.plot(gt_avg[:n], color='#2563eb', label='Ground Truth', alpha=0.8)
    ax1.plot(gen_avg[:n], color='#dc2626', label='Generated', alpha=0.8)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Mean Force Magnitude (kJ/mol/nm)')
    ax1.set_title(f'Average Force Magnitude per Frame — {system_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Distribution of all force magnitudes
    gt_flat = np.concatenate(gt_all).ravel()
    gen_flat = np.concatenate(gen_all).ravel()
    
    # Clip extreme values for better visualization
    max_val = max(np.percentile(gt_flat, 99), np.percentile(gen_flat, 99))
    
    ax2.hist(gt_flat, bins=50, range=(0, max_val), density=True, color='#2563eb', alpha=0.5, label='Ground Truth')
    ax2.hist(gen_flat, bins=50, range=(0, max_val), density=True, color='#dc2626', alpha=0.5, label='Generated')
    ax2.set_xlabel('Force Magnitude (kJ/mol/nm)')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Distribution of Atomic Force Magnitudes — {system_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved force comparison plot: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare atomic forces between two trajectories.")
    parser.add_argument("gt_xyz", help="Ground truth .xyz trajectory")
    parser.add_argument("gen_xyz", help="Generated .xyz trajectory")
    parser.add_argument("--atoms", required=True, help="Path to atom names file")
    parser.add_argument("--output", default=None, help="Output plot path")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames")
    parser.add_argument("--minimize", action="store_true", help="Minimize energy before force calculation")
    parser.add_argument("--save_npy", action="store_true", help="Save raw forces to .npy")
    args = parser.parse_args()

    atom_map = build_atom_map(args.atoms)
    forcefield_files = ['amber14-all.xml', 'implicit/gbn2.xml']
    
    gt_frames = read_xyz(args.gt_xyz)
    gen_frames = read_xyz(args.gen_xyz)
    gt_frames = gt_frames[:len(gen_frames)]  # match frame count
    
    if args.max_frames:
        gt_frames = gt_frames[:args.max_frames]
        gen_frames = gen_frames[:args.max_frames]
        
    print(f"Loaded {len(gt_frames)} GT frames and {len(gen_frames)} Generated frames (matched).")

    print("\nProcessing Ground Truth forces...")
    gt_avg, gt_all = compute_force_magnitudes(gt_frames, atom_map, forcefield_files, minimize=args.minimize)

    print("\nProcessing Generated forces...")
    gen_avg, gen_all = compute_force_magnitudes(gen_frames, atom_map, forcefield_files, minimize=args.minimize)

    if args.output is None:
        args.output = os.path.splitext(args.gen_xyz)[0] + "_force.png"
        
    system_name = os.path.splitext(os.path.basename(args.gen_xyz))[0].split('_trajectories')[0]
    plot_force_comparison(gt_avg, gt_all, gen_avg, gen_all, args.output, system_name=system_name)

    if args.save_npy:
        npy_path = os.path.splitext(args.output)[0] + "_forces.npy"
        np.save(npy_path, {"gt": gt_all, "gen": gen_all})
        print(f"Saved raw forces to {npy_path}")

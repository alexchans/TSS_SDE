"""Compare per-frame potential energy between ground truth and generated trajectories.

Uses OpenMM with Amber14 force field + implicit solvent (GBn2) for physically accurate energies.
Auto-detects atom mapping from atom names file — works for any protein.

Requires: conda activate geoTDM_m3 (or env with openmm + pdbfixer)
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
from openmm.unit import kilojoule_per_mole
from pdbfixer import PDBFixer


def read_xyz(xyz_path):
    """Read .xyz trajectory into list of (num_atoms, 3) arrays."""
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
    return frames


def build_atom_map(atom_names_path):
    """Build atom mapping from atom names file.

    Assigns residue numbers by detecting backbone N-CA-C repeats.
    Uses topology module to identify correct residue types.
    Returns list of (resname, resnum, atomname) tuples.
    """
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


def compute_energies(frames, atom_map, forcefield_files, minimize=True, max_iter=50):
    """Compute potential energy for each frame using OpenMM."""
    ff = ForceField(*forcefield_files)
    energies = []
    failed = []

    for i, coords in enumerate(tqdm(frames, desc="Computing energies", ncols=80)):
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
                sim.minimizeEnergy(maxIterations=max_iter)

            state = sim.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
            energies.append(energy)
        except Exception as e:
            energies.append(float('nan'))
            failed.append((i, str(e)))

    if failed:
        print(f"  Warning: {len(failed)} frames failed")

    return np.array(energies)


def plot_energy(gt_energies, gen_energies, output_path, system_name=''):
    """Plot ground truth vs generated energy comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))

    n = min(len(gt_energies), len(gen_energies))
    gt_energies = gt_energies[:n]
    gen_energies = gen_energies[:n]

    ax.plot(gt_energies, color='#2563eb', linewidth=0.6, alpha=0.7, label='Ground Truth')
    ax.plot(gen_energies, color='#dc2626', linewidth=0.6, alpha=0.7, label='Generated')

    ax.axhline(np.nanmean(gt_energies), color='#2563eb', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(np.nanmean(gen_energies), color='#dc2626', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Energy (kJ/mol)', fontsize=12)
    ax.set_title(
        f'Potential Energy Comparison — {system_name} (Amber14 + GBn2)\n'
        f'GT: {np.nanmean(gt_energies):.1f} ± {np.nanstd(gt_energies):.1f} | '
        f'Gen: {np.nanmean(gen_energies):.1f} ± {np.nanstd(gen_energies):.1f} kJ/mol',
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved energy plot: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare potential energy between two trajectories using OpenMM.")
    parser.add_argument("gt_xyz", help="Ground truth .xyz trajectory")
    parser.add_argument("gen_xyz", help="Generated .xyz trajectory")
    parser.add_argument("--atoms", required=True, help="Path to atom names file")
    parser.add_argument("--output", default=None, help="Output plot path")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames to evaluate")
    parser.add_argument("--no_minimize", action="store_true", help="Skip energy minimization")
    args = parser.parse_args()

    atom_map = build_atom_map(args.atoms)
    forcefield_files = ['amber14-all.xml', 'implicit/gbn2.xml']
    print(f"Atom map: {len(atom_map)} atoms, {atom_map[-1][1]} residues")
    print(f"Force field: {forcefield_files}")

    gt_frames = read_xyz(args.gt_xyz)
    gen_frames = read_xyz(args.gen_xyz)
    if args.max_frames:
        gt_frames = gt_frames[:args.max_frames]
        gen_frames = gen_frames[:args.max_frames]
    print(f"Ground truth: {len(gt_frames)} frames, Generated: {len(gen_frames)} frames")

    print("\n--- Ground Truth ---")
    gt_energies = compute_energies(gt_frames, atom_map, forcefield_files, minimize=not args.no_minimize)

    print("\n--- Generated ---")
    gen_energies = compute_energies(gen_frames, atom_map, forcefield_files, minimize=not args.no_minimize)

    print(f"\nGround Truth: {np.nanmean(gt_energies):.1f} ± {np.nanstd(gt_energies):.1f} kJ/mol")
    print(f"Generated:    {np.nanmean(gen_energies):.1f} ± {np.nanstd(gen_energies):.1f} kJ/mol")

    if args.output is None:
        base = os.path.splitext(args.gen_xyz)[0]
        args.output = base + "_energy.png"
    system_name = os.path.splitext(os.path.basename(args.gen_xyz))[0].split('_trajectories')[0]
    plot_energy(gt_energies, gen_energies, args.output, system_name=system_name)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import time
import numpy as np
import torch
import torchsde
from sde_model import GeometricSDE


def generate_trajectory(model_path, seed_frame_path, num_steps=500, dt=0.01,
                        window_size=10, diffusion_scale=1.0, k=None,
                        no_drift=False, no_diffusion=False, divergence_factor=10.0):
    """Generate a trajectory by rolling SDE integration over short windows."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, weights_only=True)
    num_residues = checkpoint['node_embedding.weight'].shape[0]

    def auto_k(n):
        return (n - 1) if n <= 50 else min(max(20, n // 2), 30)

    k_neighbors = k if k is not None else auto_k(num_residues)
    model = GeometricSDE(num_residues, k=k_neighbors).to(device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print(f"  Model: {num_residues} atoms, k={model.k} neighbors")

    # Ablation: disable drift (pure diffusion) or diffusion (pure ODE)
    if no_drift:
        model.f = lambda t, y: torch.zeros_like(y)
        print("  Drift DISABLED (f=0, diffusion-only)")
    if no_diffusion:
        model.g = lambda t, y: torch.zeros_like(y)
        print("  Diffusion DISABLED (g=0, ODE-only)")

    # Scale diffusion at inference time
    if not no_diffusion and diffusion_scale != 1.0:
        original_g = model.g
        model.g = lambda t, y: diffusion_scale * original_g(t, y)
        print(f"  Diffusion scaled to {diffusion_scale}x")

    seed_data = np.load(seed_frame_path)
    y0 = torch.from_numpy(seed_data[0].reshape(1, -1)).float().to(device)

    ts_window = torch.linspace(0, (window_size - 1) * dt, window_size).to(device)
    frames = [y0.cpu()]

    # Divergence guard. The rolling-window rollout feeds its own output back as the
    # next seed, so a drift error compounds: without this, a diverging run silently
    # writes 5000 frames of inf/NaN and still reports "Generation complete".
    seed_extent = float(np.abs(seed_data[0] - seed_data[0].mean(axis=0)).max())
    max_extent = divergence_factor * seed_extent

    with torch.no_grad():
        print(f"Generating {num_steps} frames on {device} (window_size={window_size}, "
              f"diffusion_scale={diffusion_scale})...")
        print(f"  Divergence guard: abort if any atom exceeds {max_extent:.1f} Å from "
              f"center (seed extent {seed_extent:.1f} Å x {divergence_factor})")
        steps_done = 0
        diverged_at = None
        # Time only the sampling rollout (excludes model load and file I/O). CUDA is
        # async, so synchronize before reading the clock at both ends for accuracy.
        if device.type == 'cuda':
            torch.cuda.synchronize()
        sample_start = time.perf_counter()
        while steps_done < num_steps - 1:
            model.set_edges(y0)
            chunk = torchsde.sdeint(model, y0, ts_window, method='euler', dt=dt)
            new_frames = chunk[1:]  # (window_size-1, 1, N*3)

            last = new_frames[-1].view(num_residues, 3)
            extent = (last - last.mean(dim=0)).abs().max().item()
            if not torch.isfinite(new_frames).all() or extent > max_extent:
                diverged_at = steps_done + window_size - 1
                reason = "non-finite coordinates" if not torch.isfinite(new_frames).all() \
                    else f"extent {extent:.1f} Å > {max_extent:.1f} Å"
                print(f"\n  *** DIVERGED at frame ~{diverged_at}: {reason}", flush=True)
                print(f"  *** Stopping. The drift is unstable under the current noise level.",
                      flush=True)
                print(f"  *** Try a smaller --diffusion_scale, or retrain with drift "
                      f"noise augmentation.", flush=True)
                break

            frames.append(new_frames.squeeze(1).cpu())
            y0 = new_frames[-1]
            steps_done += window_size - 1
            if steps_done % 50 < window_size:
                print(f"  Generated {min(steps_done + 1, num_steps)} / {num_steps} frames")

        if device.type == 'cuda':
            torch.cuda.synchronize()
        sample_time = time.perf_counter() - sample_start

    if diverged_at is not None:
        num_steps = min(num_steps, sum(f.shape[0] for f in frames))
        print(f"  Writing the {num_steps} finite frames produced before divergence.", flush=True)

    # Sampling timing. Per-frame is over the frames actually produced (which may be
    # fewer than requested if the run diverged), excluding the seed frame.
    frames_sampled = min(num_steps, sum(f.shape[0] for f in frames)) - 1
    per_frame = sample_time / frames_sampled if frames_sampled > 0 else float('nan')
    print(f"Sampling time: {sample_time:.3f} s total for {frames_sampled} frames "
          f"({per_frame * 1e3:.3f} ms/frame)", flush=True)

    trajectory = torch.cat(frames, dim=0)[:num_steps].numpy()
    trajectory = trajectory.reshape(num_steps, num_residues, 3)
    return trajectory


def save_xyz(trajectory, atom_names_path, output_path):
    """Save trajectory as .xyz file for visualization in PyMOL/VMD."""
    num_frames, num_atoms, _ = trajectory.shape

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

    with open(output_path, 'w') as out:
        for frame_idx in range(num_frames):
            out.write(f"{num_atoms}\n")
            out.write(f"Frame {frame_idx}\n")
            for atom_idx in range(num_atoms):
                x, y, z = trajectory[frame_idx, atom_idx]
                out.write(f"{elements[atom_idx]:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trajectories using a trained SDE model.")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of frames to generate")
    parser.add_argument("--model", type=str, default="model/ala_sde.pt", help="Path to trained model weights")
    parser.add_argument("--seed", type=str, default="data/ala.npy", help="Path to .npy data file (first frame as seed)")
    parser.add_argument("--atoms", type=str, default="data/ala_atoms.txt", help="Path to atom names file")
    parser.add_argument("--diffusion_scale", type=float, default=1.0, help="Scale diffusion by this factor (0.0=deterministic, 1.0=full noise)")
    parser.add_argument("--k", type=int, default=None, help="KNN neighbors (default: N-1, fully connected)")
    parser.add_argument("--no-drift", action="store_true", help="Disable drift (f=0), diffusion-only generation")
    parser.add_argument("--no-diffusion", action="store_true", help="Disable diffusion (g=0), deterministic ODE generation")
    parser.add_argument("--divergence_factor", type=float, default=10.0, help="Abort if any atom exceeds this multiple of the seed's extent from center (default: 10)")
    args = parser.parse_args()

    seed_name = os.path.splitext(os.path.basename(args.seed))[0]
    out_dir = os.path.join("trajectories", seed_name)
    os.makedirs(out_dir, exist_ok=True)

    new_traj = generate_trajectory(args.model, args.seed, num_steps=args.num_steps,
                                   diffusion_scale=args.diffusion_scale,
                                   k=args.k,
                                   no_drift=args.no_drift,
                                   no_diffusion=args.no_diffusion,
                                   divergence_factor=args.divergence_factor)

    suffix = ""
    if args.no_drift:
        suffix = "_no_drift"
    elif args.no_diffusion:
        suffix = "_no_diffusion"

    npy_path = os.path.join(out_dir, f"{seed_name}_trajectories_{args.num_steps}{suffix}.npy")
    np.save(npy_path, new_traj)

    xyz_path = os.path.join(out_dir, f"{seed_name}_trajectories_{args.num_steps}{suffix}.xyz")
    save_xyz(new_traj, args.atoms, xyz_path)

    print(f"Generation complete. Saved {args.num_steps} frames:")
    print(f"  {npy_path} (shape: {new_traj.shape})")
    print(f"  {xyz_path}")
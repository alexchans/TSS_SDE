"""Autoencoder latent-space comparison between ground-truth and generated trajectories.

Trains a small autoencoder on the ground-truth trajectory to learn a low-dimensional
(default 2D) latent manifold of its conformational space, then projects BOTH the
ground-truth and the generated trajectory into that space and plots them together.
This is the latent-space analog of the Ramachandran comparison: it shows whether the
SDE-generated ensemble covers the same conformational space as the reference (overlap),
explores new regions (spread beyond GT), or collapses (smaller coverage).

Inputs may be .npy (num_frames, num_atoms, 3) or .xyz. By default frames are rigid-body
superposed (Kabsch) onto the first GT frame so the latent space reflects internal
conformation rather than global translation/rotation.

An autoencoder latent basis is not identifiable: rotating the encoder output and applying
the inverse rotation in the decoder leaves the loss unchanged, so nothing pins down the
orientation, sign, or scale of the axes. With a random init those are arbitrary per run,
and two runs on identical data give clouds that are rotated/flipped relative to each
other — which makes latent plots incomparable ACROSS runs. `--seed` fixes every source of
randomness (weight init, train/val split, batch shuffling) so the same ground truth always
lands in the same latent space and plots from different runs can be compared side by side.
Note this only makes runs reproducible; it does not make the axes physically meaningful,
and comparisons within a single plot (does generated overlap GT?) remain the sound reading.
"""

import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))  # inputs are min-max normalized to [0, 1]


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def loss_function(recon_x, x, z, l2_weight=1e-3):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    l2_latent = torch.mean(z.pow(2))  # keep the latent space compact
    return recon_loss + l2_weight * l2_latent


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def read_xyz(xyz_path):
    """Read an .xyz trajectory into (num_frames, num_atoms, 3)."""
    frames = []
    with open(xyz_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        i += 2  # skip count + comment
        coords = [[float(p) for p in lines[i + j].split()[1:4]] for j in range(num_atoms)]
        frames.append(coords)
        i += num_atoms
    return np.array(frames, dtype=np.float32)


def load_coords(path, max_frames=None):
    """Load a trajectory (.npy or .xyz) as (num_frames, num_atoms, 3) float32."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
        if arr.ndim == 2:  # (frames, num_atoms*3)
            arr = arr.reshape(arr.shape[0], -1, 3)
    elif ext == '.xyz':
        arr = read_xyz(path)
    else:
        raise ValueError(f"Unsupported trajectory format: {ext} (use .npy or .xyz)")
    if max_frames is not None:
        arr = arr[:max_frames]
    return arr.astype(np.float32)


def kabsch_align(coords, ref):
    """Rigid-body superpose each frame of coords (F, N, 3) onto ref (N, 3)."""
    ref_c = ref - ref.mean(axis=0)
    mobile = coords - coords.mean(axis=1, keepdims=True)         # center each frame
    h = np.einsum('fni,nj->fij', mobile, ref_c)                  # cross-covariance (F, 3, 3)
    u, _, vt = np.linalg.svd(h)
    v, ut = np.transpose(vt, (0, 2, 1)), np.transpose(u, (0, 2, 1))
    d = np.sign(np.linalg.det(np.einsum('fij,fjk->fik', v, ut))) # fix reflections
    dmat = np.zeros((coords.shape[0], 3, 3))
    dmat[:, 0, 0] = dmat[:, 1, 1] = 1.0
    dmat[:, 2, 2] = d
    rot = np.einsum('fij,fjk,fkl->fil', v, dmat, ut)             # optimal rotation (F, 3, 3)
    return np.einsum('fni,fji->fnj', mobile, rot)                # apply R to each frame


# ---------------------------------------------------------------------------
# Train / project / plot
# ---------------------------------------------------------------------------
def train_autoencoder(data, latent_dim, device, epochs, batch_size, lr,
                      val_ratio=0.2, patience=10, seed=0):
    """Train an AE on `data` (num_frames, input_dim). Returns the best model.

    `seed` fixes the three sources of run-to-run variation — the train/val split, the
    batch shuffling, and the weight init — so the learned latent basis is reproducible.
    """
    input_dim = data.shape[1]
    tensor = torch.tensor(data, dtype=torch.float32)
    train_size = int((1 - val_ratio) * len(tensor))
    split_gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        TensorDataset(tensor), [train_size, len(tensor) - train_size],
        generator=split_gen
    )
    shuffle_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=shuffle_gen)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Seed immediately before constructing the model so weight init is deterministic
    # regardless of how much RNG the split/loader above consumed.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = AutoEncoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, best_state, stale = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = loss_function(recon, batch, z)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, z = model(batch)
                val_loss += loss_function(recon, batch, z).item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        print(f"Epoch {epoch + 1:4d} | train {train_loss:.5f} | val {val_loss:.5f}", flush=True)

        if val_loss < best_val:
            best_val, best_state, stale = val_loss, {k: v.detach().cpu().clone()
                                                     for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch + 1} (best val {best_val:.5f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def project(model, data, device):
    """Encode `data` (num_frames, input_dim) to latent coordinates and recon MSE."""
    model.eval()
    tensor = torch.tensor(data, dtype=torch.float32, device=device)
    recon, z = model(tensor)
    recon_mse = F.mse_loss(recon, tensor, reduction='mean').item()
    return z.cpu().numpy(), recon_mse


def plot_latent(z_gt, z_gen, output_path, system_name=''):
    """Side-by-side GT/generated latent densities plus an overlay scatter."""
    both = np.concatenate([z_gt, z_gen], axis=0)
    xlim = (both[:, 0].min(), both[:, 0].max())
    ylim = (both[:, 1].min(), both[:, 1].max())

    fig, (ax_gt, ax_gen, ax_ov) = plt.subplots(1, 3, figsize=(20, 6))
    if system_name:
        fig.suptitle(f'Autoencoder latent space — {system_name}', fontsize=15, fontweight='bold')

    rng = [list(xlim), list(ylim)]
    for ax, z, title in [(ax_gt, z_gt, f'Ground Truth ({len(z_gt)})'),
                         (ax_gen, z_gen, f'Generated ({len(z_gen)})')]:
        _, _, _, im = ax.hist2d(z[:, 0], z[:, 1], bins=80, range=rng, cmap='inferno', cmin=1)
        ax.set_title(title, fontsize=13, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Density', shrink=0.8)

    ax_ov.scatter(z_gt[:, 0], z_gt[:, 1], s=4, alpha=0.3, c='tab:gray', label='Ground Truth')
    ax_ov.scatter(z_gen[:, 0], z_gen[:, 1], s=4, alpha=0.3, c='tab:orange', label='Generated')
    ax_ov.set_title('Overlay', fontsize=13, fontweight='bold')
    ax_ov.legend(markerscale=3, framealpha=0.9)

    for ax in (ax_gt, ax_gen, ax_ov):
        ax.set_xlabel('AE latent 1', fontsize=12)
        ax.set_ylabel('AE latent 2', fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.97) if system_name else None)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved latent-space plot: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder latent-space comparison of trajectories.")
    parser.add_argument("xyz", help="Generated trajectory (.npy or .xyz)")
    parser.add_argument("--gt", required=True, help="Ground-truth trajectory (.npy or .xyz); the AE trains on this")
    parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension (only 2 is plotted)")
    parser.add_argument("--epochs", type=int, default=500, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_frames", type=int, default=None, help="Cap frames loaded from each trajectory")
    parser.add_argument("--no-align", action="store_true", help="Skip Kabsch superposition before encoding")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the AE's weight init, train/val split, and batch shuffling. Fixed by default so the same ground truth always maps to the same latent space and plots from separate runs are comparable")
    parser.add_argument("--output", default=None, help="Output plot path (default: <input>_latent.png)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"######## Using {device} ########", flush=True)

    gt = load_coords(args.gt, args.max_frames)
    gen = load_coords(args.xyz, args.max_frames)
    print(f"GT: {gt.shape}  |  Generated: {gen.shape}", flush=True)
    if gt.shape[1] != gen.shape[1]:
        raise ValueError(f"Atom count mismatch: GT has {gt.shape[1]}, generated has {gen.shape[1]}")

    # Align both to the same reference (first GT frame) so the latent space is
    # about internal conformation, not rigid-body pose.
    if not args.no_align:
        ref = gt[0]
        gt = kabsch_align(gt, ref)
        gen = kabsch_align(gen, ref)
        print("Applied Kabsch superposition to GT frame 0", flush=True)

    gt = gt.reshape(gt.shape[0], -1)
    gen = gen.reshape(gen.shape[0], -1)

    # Min-max normalize to [0, 1] using GT statistics; apply the same map to generated.
    dmin, dmax = gt.min(), gt.max()
    scale = (dmax - dmin) if dmax > dmin else 1.0
    gt_n = (gt - dmin) / scale
    gen_n = (gen - dmin) / scale

    print(f"Training autoencoder (input_dim={gt_n.shape[1]}, latent_dim={args.latent_dim}) on GT...", flush=True)
    model = train_autoencoder(gt_n, args.latent_dim, device,
                              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                              seed=args.seed)

    z_gt, mse_gt = project(model, gt_n, device)
    z_gen, mse_gen = project(model, gen_n, device)
    print(f"Reconstruction MSE — GT: {mse_gt:.5f} | Generated: {mse_gen:.5f}", flush=True)
    print(f"  (generated MSE >> GT MSE means the generated ensemble leaves the learned GT manifold)", flush=True)

    if args.latent_dim != 2:
        print(f"latent_dim={args.latent_dim}; plotting first 2 dimensions only.", flush=True)
        z_gt, z_gen = z_gt[:, :2], z_gen[:, :2]

    if args.output is None:
        args.output = os.path.splitext(args.xyz)[0] + "_latent.png"
    stem = os.path.splitext(os.path.basename(args.xyz))[0]
    system_name = stem.split('_trajectories')[0]
    # Flag ablation rollouts in the title — the plot filename alone is easy to mix up
    # when comparing against the stochastic run side by side.
    if stem.endswith('_no_diffusion'):
        system_name += " (no diffusion, ODE only)"
    elif stem.endswith('_no_drift'):
        system_name += " (no drift, diffusion only)"
    plot_latent(z_gt, z_gen, args.output, system_name=system_name)

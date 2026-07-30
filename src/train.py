import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy
import argparse
import warnings
warnings.filterwarnings('ignore', message='.*epoch parameter.*scheduler.step.*')
import torch
import torch.distributed as dist
import torchsde
from sde_model import GeometricSDE
from utils.dataset import ProteinTrajectoryDataset
from utils.topology import build_bonds_from_atom_names

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
parser.add_argument('--data', type=str, default='data/ala.npy', help='Path to coordinate .npy file')
parser.add_argument('--atoms', type=str, default='data/ala_atoms.txt', help='Path to atom names file')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
parser.add_argument('--max_frames', type=int, default=50000, help='Max training frames (default: 50000)')
parser.add_argument('--k', type=int, default=None, help='KNN neighbors per atom (default: N-1, i.e. fully connected)')
parser.add_argument('--nll_weight', type=float, default=0.1, help='Weight on the Euler-Maruyama transition NLL term')
args = parser.parse_args()

# Distributed setup
dist.init_process_group(backend='nccl')
rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)

# Configuration
data_path = args.data
atom_names_path = args.atoms
data_name = os.path.splitext(os.path.basename(data_path))[0]  # e.g. 'ubiq' from 'data/ubiq.npy'
model_best_path = f"model/{data_name}_sde.pt"
model_final_path = f"model/{data_name}_sde_final.pt"
os.makedirs('model', exist_ok=True)
window_size = 10
dt = 0.01
grad_accum_steps = 1
# Rollout loss weights. Reused for checkpoint selection, so keep them in one place.
W_POS, W_VEL, W_BOND, W_CLASH = 0.45, 0.30, 0.15, 0.10

# Dataset & DataLoader
dataset = ProteinTrajectoryDataset(data_path, window_size=window_size, max_frames=args.max_frames)
num_residues = dataset.num_residues
if rank == 0:
    print(f"Detected {num_residues} residues from {data_path}", flush=True)
    print(f"Training on {world_size} GPUs, gradient accumulation: {grad_accum_steps}", flush=True)
    print(f"Single-phase SDE training (drift + diffusion jointly): epochs 0-{args.epochs-1}", flush=True)

sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, num_replicas=world_size, rank=rank, shuffle=True
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True
)

# Geometric loss helpers

def make_bond_tensors(bond_defs, device):
    """Convert bond definitions to index tensors and ideal length tensor."""
    bond_i = torch.tensor([b[0] for b in bond_defs], dtype=torch.long, device=device)
    bond_j = torch.tensor([b[1] for b in bond_defs], dtype=torch.long, device=device)
    bond_lengths = torch.tensor([b[2] for b in bond_defs], dtype=torch.float32, device=device)
    bonded_set = set()
    for i, j, _ in bond_defs:
        bonded_set.add((min(i, j), max(i, j)))
    return bond_i, bond_j, bond_lengths, bonded_set

def make_nonbonded_pairs(num_atoms, bonded_set, device):
    """Build non-bonded pair indices, excluding 1-2 and 1-3 bonded neighbors."""
    adj = {i: set() for i in range(num_atoms)}
    for (a, b) in bonded_set:
        adj[a].add(b)
        adj[b].add(a)

    exclude = set(bonded_set)
    for a in range(num_atoms):
        for b in adj[a]:
            for c in adj[b]:
                if c != a:
                    exclude.add((min(a, c), max(a, c)))

    pairs_i, pairs_j = [], []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if (i, j) not in exclude:
                pairs_i.append(i)
                pairs_j.append(j)
    return (torch.tensor(pairs_i, dtype=torch.long, device=device),
            torch.tensor(pairs_j, dtype=torch.long, device=device))

def bond_length_loss(coords, bond_i, bond_j, ideal_lengths):
    """MSE between predicted and ideal bond lengths."""
    diff = coords[:, :, bond_i] - coords[:, :, bond_j]
    dists = torch.norm(diff, dim=-1)
    return torch.nn.functional.mse_loss(dists, ideal_lengths.unsqueeze(0).unsqueeze(0).expand_as(dists))

def clash_loss(coords, nb_i, nb_j, min_dist=1.0):
    """Penalize non-bonded atom pairs closer than min_dist."""
    diff = coords[:, :, nb_i] - coords[:, :, nb_j]
    dists = torch.norm(diff, dim=-1)
    violations = torch.relu(min_dist - dists)
    return (violations ** 2).mean()

# Model & Optimizer
def auto_k(n_atoms):
    """Auto-select KNN neighbors based on molecule size.
    ≤50 atoms: fully connected (N-1) — critical for small peptides.
    >50 atoms: 50% connectivity, capped at 30 for tractability.
    """
    if n_atoms <= 50:
        return n_atoms - 1
    return min(max(20, n_atoms // 2), 30)

k_neighbors = args.k if args.k is not None else auto_k(num_residues)
model = GeometricSDE(num_residues, k=k_neighbors).to(device)
if rank == 0:
    print(f"KNN neighbors k={model.k} (out of {num_residues - 1} possible, {'auto' if args.k is None else 'manual'})", flush=True)
peak_lr = 5e-4
warmup_epochs = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=1e-4)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
)

# EMA model
ema_decay = 0.999
ema_model = copy.deepcopy(model)
ema_model.eval()
for p in ema_model.parameters():
    p.requires_grad_(False)

@torch.no_grad()
def update_ema():
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

# Sync parameters across GPUs
for param in model.parameters():
    dist.broadcast(param.data, src=0)
for param in ema_model.parameters():
    dist.broadcast(param.data, src=0)

ts = torch.linspace(0, (window_size - 1) * dt, window_size).to(device)

# Auto-detect bonds from topology
bond_defs, residue_info = build_bonds_from_atom_names(atom_names_path)
bond_i, bond_j, ideal_lengths, bonded_set = make_bond_tensors(bond_defs, device)
nb_i, nb_j = make_nonbonded_pairs(num_residues, bonded_set, device)

if rank == 0:
    print(f"Auto-detected {len(bond_defs)} bonds from topology ({atom_names_path})", flush=True)
    for i, j, d in bond_defs:
        print(f"  Bond {i}-{j}: {d:.3f} Å")
    print(f"Non-bonded pairs (for clash): {len(nb_i)} pairs", flush=True)

# Training loop
best_loss = float('inf')
# EMA of loss for the dynamic skip threshold. Seeded from the first finite batch
# rather than a constant: the NLL term's scale is data-dependent (it carries a
# 1/(2·g²·dt) factor), so a hardcoded seed can put every batch over the skip
# threshold — and since the EMA only advances on batches that are *not* skipped,
# that state is self-locking and would stall training at epoch 0.
running_loss_avg = None
for epoch in range(args.epochs):
    sampler.set_epoch(epoch)
    epoch_loss = 0.0
    epoch_pos_loss = 0.0
    epoch_vel_loss = 0.0
    epoch_bond_loss = 0.0
    epoch_clash_loss = 0.0
    epoch_nll_loss = 0.0
    epoch_g_mean = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        y0 = batch[:, 0, :]
        B, T, D = batch.shape

        model.set_edges(y0)  # cache KNN graph for all Euler steps

        # Drift-only rollout (g=0): rollout accuracy + geometry. Noise-free, so
        # none of these terms can push the diffusion toward zero — an MSE against
        # a stochastic rollout pays bias² + Var, and g only ever adds variance.
        model.g = lambda t, y: torch.zeros_like(y)
        pred_trajectory = torchsde.sdeint(model, y0, ts, method='euler', dt=dt)
        del model.g  # restore class method
        pred_trajectory = pred_trajectory.permute(1, 0, 2)

        pos_loss = torch.nn.functional.mse_loss(pred_trajectory, batch)

        pred_vel = pred_trajectory[:, 1:, :] - pred_trajectory[:, :-1, :]
        true_vel = batch[:, 1:, :] - batch[:, :-1, :]
        vel_loss = torch.nn.functional.mse_loss(pred_vel, true_vel)

        pred_coords = pred_trajectory.view(B, T, num_residues, 3)
        b_loss = bond_length_loss(pred_coords, bond_i, bond_j, ideal_lengths)
        c_loss = clash_loss(pred_coords, nb_i, nb_j, min_dist=1.0)

        # Euler-Maruyama transition NLL, teacher-forced on consecutive GT frames.
        # x_{k+1} | x_k ~ N(x_k + f·dt, g²·dt), so minimizing
        #     (dx − f·dt)² / (2·g²·dt) + log g
        # fits f to the conditional mean of the increments and g to the residual
        # scale — the motion f cannot predict. This identifies g even with the
        # drift trainable: g→0 blows up the quadratic term, log g caps it above.
        # Variance-matching could not do this (a trainable drift absorbs the
        # whole variance target), which is why the old code had to freeze f.
        y_flat = batch[:, :-1, :].reshape(B * (T - 1), D)
        dx_flat = (batch[:, 1:, :] - batch[:, :-1, :]).reshape(B * (T - 1), D)
        t_flat = ts[:-1].unsqueeze(0).expand(B, -1).reshape(-1)

        f_pred = model.f(t_flat, y_flat)
        g_out = model.g(t_flat, y_flat)
        resid = dx_flat - f_pred * dt
        nll = (resid ** 2 / (2.0 * g_out ** 2 * dt) + torch.log(g_out)).mean()

        model._cached_edges = None  # free cache
        g_mean = g_out.mean()

        # g appears only in nll, so nll_weight rescales its gradient but not its
        # optimum; it trades f's rollout fit against its one-step conditional mean.
        rollout_loss = (W_POS * pos_loss + W_VEL * vel_loss
                        + W_BOND * b_loss + W_CLASH * c_loss)
        loss = rollout_loss + args.nll_weight * nll

        loss_val = loss.item()
        if running_loss_avg is None and torch.isfinite(loss):
            running_loss_avg = abs(loss_val)
        # abs(): log g makes the NLL, and hence the total loss, legitimately negative.
        skip_threshold = max(5.0 * abs(running_loss_avg or 0.0), 1.0)
        skip_local = not torch.isfinite(loss) or loss_val > skip_threshold
        # Synchronize skip decision across all ranks to prevent NCCL deadlock
        skip_tensor = torch.tensor([1.0 if skip_local else 0.0], device=device)
        dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
        skip_batch = skip_tensor.item() > 0.5
        if skip_batch:
            optimizer.zero_grad()
            continue

        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Manual all-reduce (torchsde bypasses nn.Module.forward, so DDP can't be used)
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(world_size)

            # Skip update if any gradients are NaN/inf
            grad_ok = all(
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            )
            if grad_ok:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                update_ema()
            optimizer.zero_grad()

        running_loss_avg = 0.95 * running_loss_avg + 0.05 * loss_val
        epoch_loss += loss_val
        epoch_pos_loss += pos_loss.item()
        epoch_vel_loss += vel_loss.item()
        epoch_bond_loss += b_loss.item()
        epoch_clash_loss += c_loss.item()
        epoch_nll_loss += nll.item()
        epoch_g_mean += g_mean.item()
        num_batches += 1

    avg_loss = epoch_loss / max(num_batches, 1)
    avg_pos = epoch_pos_loss / max(num_batches, 1)
    avg_vel = epoch_vel_loss / max(num_batches, 1)
    avg_bond = epoch_bond_loss / max(num_batches, 1)
    avg_clash = epoch_clash_loss / max(num_batches, 1)
    avg_nll = epoch_nll_loss / max(num_batches, 1)
    avg_g = epoch_g_mean / max(num_batches, 1)

    if num_batches == 0:
        # NaN recovery: restore model from EMA (last known good state)
        model.load_state_dict(ema_model.state_dict())
        optimizer.zero_grad()
        # Sync restored weights across GPUs
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        if rank == 0:
            print(f"Epoch {epoch} | WARNING: All batches skipped — restored model from EMA checkpoint.", flush=True)

    # Checkpoint on rollout quality alone. The NLL's log g term has a large dynamic
    # range and can go negative, so folding it in would make "best" track the
    # diffusion fit rather than the trajectory the model actually generates.
    avg_rollout = (W_POS * avg_pos + W_VEL * avg_vel
                   + W_BOND * avg_bond + W_CLASH * avg_clash)

    if rank == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} [SDE] | Loss: {avg_loss:.6f} (rollout: {avg_rollout:.6f}, pos: {avg_pos:.6f}, vel: {avg_vel:.6f}, bond: {avg_bond:.6f}, clash: {avg_clash:.6f}, nll: {avg_nll:.6f}, g: {avg_g:.4f}) | LR: {lr:.2e}", flush=True)
        # Only consider the checkpoint on a real epoch. A fully-skipped epoch has
        # num_batches==0, so every avg_* is 0/1 = 0 and avg_rollout == 0.0 — which
        # would always beat best_loss, pin it to 0.0, and overwrite the checkpoint
        # with a half-trained EMA that no later epoch can ever displace.
        if num_batches > 0 and avg_rollout < best_loss:
            best_loss = avg_rollout
            torch.save(ema_model.state_dict(), model_best_path)

    scheduler.step()

if rank == 0:
    torch.save(ema_model.state_dict(), model_final_path)
    print(f"Training complete. Best loss: {best_loss:.6f}", flush=True)

dist.destroy_process_group()
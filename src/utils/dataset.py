import numpy as np
import torch
from torch.utils.data import Dataset

class ProteinTrajectoryDataset(Dataset):
    def __init__(self, npy_file_path, window_size=10, max_frames=None):
        raw = np.load(npy_file_path)
        if max_frames is not None:
            raw = raw[:max_frames]
        # Raw shape: (Total_Frames, Num_Residues, 3)
        # Flatten to: (Total_Frames, Num_Residues * 3) for the SDE model
        self.num_residues = raw.shape[1]
        self.data = torch.from_numpy(raw.reshape(raw.shape[0], -1)).float()
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # Returns a chunk of consecutive frames: (window_size, Num_Residues * 3)
        return self.data[idx : idx + self.window_size]

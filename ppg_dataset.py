# ppg_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PPGDataset(Dataset):
    """
    Loads PPG segments and labels from a .npz file created by create_new_dataset.py.
    Expects keys: 'ppg' (N, 1000), 'labels' (N,)
    """
    def __init__(self, npz_path):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        data = np.load(npz_path)
        self.ppg = data["ppg"]      # shape (N, 1000)
        self.labels = data["labels"]  # shape (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.ppg[idx]          # (1000,)
        y = int(self.labels[idx])  # scalar
        # Add channel dimension: (1, 1000) for Conv1d
        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.tensor(y).long()
        return x, y


def build_dataloaders(
    base_dir: str,
    batch_size: int = 128,
    num_workers: int = 0,
):
    """
    Convenience function to create train / val / test dataloaders.
    base_dir should be the project root containing data_new/.
    """
    data_new_dir = os.path.join(base_dir, "data_new")
    train_path = os.path.join(data_new_dir, "train.npz")
    val_path   = os.path.join(data_new_dir, "val.npz")
    test_path  = os.path.join(data_new_dir, "test.npz")

    train_ds = PPGDataset(train_path)
    val_ds   = PPGDataset(val_path)
    test_ds  = PPGDataset(test_path)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader

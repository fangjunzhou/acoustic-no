from torch.utils.data import Dataset
import numpy as np
import pathlib
import logging
import torch


class AcousticDataset(Dataset):
    def __init__(self, data_dir: pathlib.Path, depth: int = 8):
        # Initialize the dataset with the directory containing the checkpoints.
        self.data_dir = data_dir
        self.depth = depth
        self.checkpoints = sorted(data_dir.glob("*.npz"))
        self.num_checkpoints = len(self.checkpoints)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Found {self.num_checkpoints} checkpoints in {data_dir}.")
        if self.num_checkpoints == 0:
            raise ValueError(f"No checkpoints found in {data_dir}.")
        # Concatenate the checkpoints into a single array.
        self.p_grid = []
        self.v_grid = []
        self.alpha_grid = []
        for checkpoint in self.checkpoints:
            data = np.load(checkpoint)
            self.p_grid.append(data["pressure"])
            self.v_grid.append(data["velocity"])
            self.alpha_grid.append(data["alpha"])
        self.p_grid = np.concatenate(self.p_grid, axis=0)
        self.v_grid = np.concatenate(self.v_grid, axis=0)
        self.alpha_grid = np.concatenate(self.alpha_grid, axis=0)
        self.length = len(self.p_grid)

    def __len__(self):
        # Return the number of instances in the dataset.
        return self.length - self.depth

    def __getitem__(self, idx):
        # Get the pressure, velocity, and alpha grids for the current instance.
        p = self.p_grid[idx : idx + self.depth]
        v = self.v_grid[idx : idx + self.depth]
        a = self.alpha_grid[idx : idx + self.depth]
        # Convert to PyTorch tensors.
        p = torch.tensor(p, dtype=torch.float32)
        v = torch.tensor(v, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.float32)
        # Compose the input and output tensors.
        depth, height, width = p.shape
        v = v.permute(0, 3, 1, 2).reshape(-1, height, width)
        x = torch.concatenate(
            [
                p[0].reshape(1, height, width),
                v,
                a,
            ],
            dim=0,
        )
        return {
            "x": x,
            "y": p,
            "v": v,
            "a": a,
        }

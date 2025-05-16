from torch.utils.data import Dataset
import numpy as np
import pathlib
import logging
import torch


class AcousticDataset(Dataset):
    def __init__(self, data_dir: pathlib.Path, depth: int = 8):
        # Initialize the dataset with the directory containing the checkpoints.
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.depth = depth
        self.scenes = []
        # Get all scenes in the directory.
        for scene in data_dir.iterdir():
            if scene.is_dir():
                # Read all checkpoints in the scene directory.
                checkpoint_dir = scene / "checkpoints"
                checkpoints = sorted(checkpoint_dir.glob("*.npz"))
                self.logger.info(
                    f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}."
                )
                p_grid = []
                v_grid = []
                a_grid = []
                # Load the checkpoints.
                for checkpoint in checkpoints:
                    data = np.load(checkpoint)
                    p_grid.append(data["pressure"])
                    v_grid.append(data["velocity"])
                    a_grid.append(data["alpha"])
                p_grid = np.concatenate(p_grid, axis=0)
                v_grid = np.concatenate(v_grid, axis=0)
                a_grid = np.concatenate(a_grid, axis=0)
                length = len(p_grid)
                # Append the scene and its checkpoints to the list.
                self.scenes.append((length, p_grid, v_grid, a_grid))
        # Calculate the total length of the dataset.
        self.length = sum(length - depth + 1 for length, _, _, _ in self.scenes)

    def __len__(self):
        # Return the number of instances in the dataset.
        return self.length

    def __getitem__(self, idx):
        # Get the scene index and the offset within the scene.
        scene_idx = 0
        while idx >= self.scenes[scene_idx][0] - self.depth + 1:
            idx -= self.scenes[scene_idx][0] - self.depth + 1
            scene_idx += 1
        # Get the scene data.
        length, p_grid, v_grid, a_grid = self.scenes[scene_idx]
        # Get the pressure, velocity, and alpha grids for the current instance.
        p = p_grid[idx : idx + self.depth]
        v = v_grid[idx : idx + self.depth]
        a = a_grid[idx : idx + self.depth]
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

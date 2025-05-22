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
        self.samples = []                # List of (file_path, frame_idx)
        # Get all scenes in the directory.
        for scene in data_dir.iterdir():
            if scene.is_dir():
                # Read all checkpoints in the scene directory.
                checkpoint_dir = scene / "checkpoints"
                checkpoints = sorted(checkpoint_dir.glob("*.npz"))
                self.logger.info(
                    f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}."
                )
                # Load the checkpoints.
                for checkpoint in checkpoints:
                    # Calculate the number of frames in the checkpoint.
                    with np.load(checkpoint) as data:
                        num_frames = data["pressure"].shape[0]
                    # Append frame indices to the samples list.
                    for i in range(num_frames - depth + 1):
                        self.samples.append((checkpoint, i))

    def __len__(self):
        # Return the number of instances in the dataset.
        return len(self.samples)

    def __getitem__(self, idx):
        # Get the file path and frame index of the sample.
        file_path, frame_idx = self.samples[idx]
         # Get the pressure, velocity, and alpha grids for the current instance.
        with np.load(file_path) as data:
            p = data["pressure"][frame_idx : frame_idx + self.depth]
            v = data["velocity"][frame_idx : frame_idx + self.depth]
            a = data["alpha"][frame_idx : frame_idx + self.depth]
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

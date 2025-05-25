from torch.utils.data import Dataset
import numpy as np
import pathlib
import logging
import torch


SAMPLE_PER_SCENE = 128


class ShuffledAcousticDataset(Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        num_chunks: int,
        depth: int = 8,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.depth = depth
        self.cache_chunk_idx = -1
        self.cache_chunk_data = None

        # Temp chunk storage.
        if not pathlib.Path(".temp").exists():
            pathlib.Path(".temp").mkdir()

        self.length = 0
        self.num_scenes = len([scene for scene in data_dir.iterdir() if scene.is_dir()])
        for chunk in range(num_chunks):
            enough_samples = True
            np.random.seed(0)
            # All pressure, velocity, and alpha grids will be stored in these lists.
            chunk_p_grid = []
            chunk_v_grid = []
            chunk_a_grid = []
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
                    # Shuffle the data.
                    indices = np.random.permutation(length - depth + 1)
                    # Sample a fixed number of samples from the shuffled data.
                    indices = indices[
                        chunk * SAMPLE_PER_SCENE : (chunk + 1) * SAMPLE_PER_SCENE
                    ]
                    # Ensure we have enough samples.
                    if len(indices) < SAMPLE_PER_SCENE:
                        enough_samples = False
                        break
                    for idx in indices:
                        # Get the pressure, velocity, and alpha grids for the current instance.
                        p_grid_chunk = p_grid[idx : idx + depth]
                        v_grid_chunk = v_grid[idx : idx + depth]
                        a_grid_chunk = a_grid[idx : idx + depth]
                        # Append the chunk.
                        chunk_p_grid.append(p_grid_chunk)
                        chunk_v_grid.append(v_grid_chunk)
                        chunk_a_grid.append(a_grid_chunk)
            if not enough_samples:
                break
            # Concatenate all chunks for the current chunk.
            chunk_p_grid = np.concatenate(chunk_p_grid, axis=0)
            chunk_v_grid = np.concatenate(chunk_v_grid, axis=0)
            chunk_a_grid = np.concatenate(chunk_a_grid, axis=0)
            # Shuffle the chunk data.
            indices = np.random.permutation(len(chunk_p_grid))
            chunk_p_grid = chunk_p_grid[indices]
            chunk_v_grid = chunk_v_grid[indices]
            chunk_a_grid = chunk_a_grid[indices]
            # Save the chunk to a file.
            chunk_file = pathlib.Path(f".temp/chunk_{chunk}.npz")
            np.savez(
                chunk_file,
                pressure=chunk_p_grid,
                velocity=chunk_v_grid,
                alpha=chunk_a_grid,
            )
            self.length += len(chunk_p_grid)

    def __len__(self):
        # Return the number of instances in the dataset.
        return self.length

    def __getitem__(self, idx):
        # Calculate the chunk index and the index within the chunk.
        chunk_idx = idx // (SAMPLE_PER_SCENE * self.num_scenes)
        idx_in_chunk = idx % (SAMPLE_PER_SCENE * self.num_scenes)
        # If the chunk is not cached, load it.
        if self.cache_chunk_idx != chunk_idx:
            # Load the chunk data.
            chunk_file = pathlib.Path(f".temp/chunk_{chunk_idx}.npz")
            data = np.load(chunk_file)
            self.cache_chunk_idx = chunk_idx
            self.cache_chunk_data = data
        else:
            # Use the cached chunk data.
            data = self.cache_chunk_data
        assert data is not None, "Chunk data should not be None."
        p_grid = data["pressure"]
        v_grid = data["velocity"]
        a_grid = data["alpha"]
        # Get the pressure, velocity, and alpha grids for the current instance.
        p = p_grid[idx_in_chunk]
        v = v_grid[idx_in_chunk]
        a = a_grid[idx_in_chunk]
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

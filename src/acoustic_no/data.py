from torch.utils.data import Dataset
import numpy as np
import pathlib
import logging
import torch
from tqdm import tqdm
import json
import multiprocessing as mp
from functools import partial


def process_chunk(chunk, data_dir, output_dir, samples_per_scene, depth, logger):
    """
    Process a single data chunk.
    
    Args:
        chunk: Chunk index to process
        data_dir: Directory containing scene data
        output_dir: Directory to save preprocessed data
        depth: Depth of the sequence for each sample
        logger: Logger instance
        
    Returns:
        Tuple of (success, num_samples) where success is a boolean indicating if enough samples 
        were found and num_samples is the number of samples in the chunk
    """
    enough_samples = True
    np.random.seed(0)  # Use consistent seed for reproducibility
    # All pressure, velocity, and alpha grids will be stored in these lists
    chunk_p_grid = []
    chunk_v_grid = []
    chunk_a_grid = []
    
    # Get all scenes in the directory
    for scene in data_dir.iterdir():
        if scene.is_dir():
            # Read all checkpoints in the scene directory
            checkpoint_dir = scene / "checkpoints"
            checkpoints = sorted(checkpoint_dir.glob("*.npz"))
            
            p_grid = []
            v_grid = []
            a_grid = []
            
            # Load the checkpoints
            for checkpoint in checkpoints:
                data = np.load(checkpoint)
                p_grid.append(data["pressure"])
                v_grid.append(data["velocity"])
                a_grid.append(data["alpha"])
            
            p_grid = np.concatenate(p_grid, axis=0)
            v_grid = np.concatenate(v_grid, axis=0)
            a_grid = np.concatenate(a_grid, axis=0)
            length = len(p_grid)
            
            # Shuffle the data
            indices = np.random.permutation(length - depth + 1)
            # Sample a fixed number of samples from the shuffled data
            indices = indices[
                chunk * samples_per_scene : (chunk + 1) * samples_per_scene
            ]
            
            # Ensure we have enough samples
            if len(indices) < samples_per_scene:
                enough_samples = False
                break
            
            for idx in indices:
                # Get the pressure, velocity, and alpha grids for the current instance
                p_grid_chunk = p_grid[idx : idx + depth]
                v_grid_chunk = v_grid[idx : idx + depth]
                a_grid_chunk = a_grid[idx : idx + depth]
                # Append the chunk
                chunk_p_grid.append(p_grid_chunk)
                chunk_v_grid.append(v_grid_chunk)
                chunk_a_grid.append(a_grid_chunk)
    
    if not enough_samples:
        return (False, 0)
        
    # Concatenate all chunks for the current chunk
    chunk_p_grid = np.array(chunk_p_grid)
    chunk_v_grid = np.array(chunk_v_grid)
    chunk_a_grid = np.array(chunk_a_grid)
    
    # Shuffle the chunk data
    indices = np.random.permutation(len(chunk_p_grid))
    chunk_p_grid = chunk_p_grid[indices]
    chunk_v_grid = chunk_v_grid[indices]
    chunk_a_grid = chunk_a_grid[indices]
    
    # Save the chunk to a file in output directory
    chunk_file = output_dir / f"chunk_{chunk}.npz"
    np.savez(
        chunk_file,
        pressure=chunk_p_grid,
        velocity=chunk_v_grid,
        alpha=chunk_a_grid,
    )
    
    return (True, len(chunk_p_grid))


def preprocess_acoustic_dataset(
    data_dir: pathlib.Path,
    output_dir: pathlib.Path,
    num_chunks: int,
    samples_per_scene: int = 64,
    depth: int = 8,
    num_workers: int = 8,
) -> None:
    """
    Preprocess acoustic dataset by loading checkpoints, sampling, and shuffling data.
    Saves the processed dataset to the specified output directory.
    
    Args:
        data_dir: Directory containing scene data
        output_dir: Directory to save preprocessed data
        num_chunks: Number of chunks to generate
        depth: Depth of the sequence for each sample
        num_workers: Number of worker processes to use (defaults to CPU count)
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count number of scenes
    num_scenes = len([scene for scene in data_dir.iterdir() if scene.is_dir()])
    logger.info(f"Found {num_scenes} scenes in {data_dir}")
    
    # Determine number of workers (default to CPU count if not specified)
    if num_workers is None:
        num_workers = mp.cpu_count()
    num_workers = min(num_workers, num_chunks)  # Don't use more workers than chunks
    logger.info(f"Using {num_workers} worker processes")
    
    # Dataset metadata
    metadata = {
        "depth": depth,
        "num_chunks": 0,
        "num_scenes": num_scenes,
        "total_samples": 0
    }
    
    # Create a partial function with fixed arguments
    process_chunk_partial = partial(
        process_chunk, 
        data_dir=data_dir, 
        output_dir=output_dir, 
        samples_per_scene=samples_per_scene,
        depth=depth,
        logger=logger
    )
    
    # Process chunks in parallel using multiprocessing
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_chunk_partial, range(num_chunks)),
            total=num_chunks,
            desc="Processing chunks"
        ))
    
    # Update metadata based on results
    for success, num_samples in results:
        if success:
            metadata["num_chunks"] += 1
            metadata["total_samples"] += num_samples
    
    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    logger.info(f"Preprocessed dataset saved to {output_dir}")
    logger.info(f"Total chunks: {metadata['num_chunks']}, total samples: {metadata['total_samples']}")


class ShuffledAcousticDataset(Dataset):
    def __init__(
        self,
        dataset_dir: pathlib.Path,
        samples_per_scene: int = 64,
    ) -> None:
        """
        Load a preprocessed acoustic dataset from the specified directory.
        
        Args:
            dataset_dir: Directory containing the preprocessed dataset
        """
        self.logger = logging.getLogger(__name__)
        self.dataset_dir = dataset_dir
        self.cache_chunk_idx = -1
        self.cache_chunk_data = None
        self.samples_per_scene = samples_per_scene
        
        # Load metadata
        try:
            with open(dataset_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
            
            self.depth = metadata["depth"]
            self.num_chunks = metadata["num_chunks"]
            self.num_scenes = metadata["num_scenes"]
            self.length = metadata["total_samples"]
            
            self.logger.info(f"Loaded dataset from {dataset_dir} with {self.length} samples")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid dataset directory: {dataset_dir}. {str(e)}")

    def __len__(self):
        # Return the number of instances in the dataset.
        return self.length

    def __getitem__(self, idx):
        # Calculate the chunk index and the index within the chunk.
        chunk_idx = idx // (self.samples_per_scene * self.num_scenes)
        idx_in_chunk = idx % (self.samples_per_scene * self.num_scenes)
        # If the chunk is not cached, load it.
        if self.cache_chunk_idx != chunk_idx:
            # Load the chunk data.
            chunk_file = self.dataset_dir / f"chunk_{chunk_idx}.npz"
            data = np.load(chunk_file)
            p_grid = data["pressure"]
            v_grid = data["velocity"]
            a_grid = data["alpha"]
            self.cache_chunk_idx = chunk_idx
            self.cache_chunk_data = (
                p_grid,
                v_grid,
                a_grid,
            )
        else:
            # Use the cached chunk data.
            p_grid, v_grid, a_grid = self.cache_chunk_data
        # Ensure the data is not None.
        assert p_grid is not None, "Pressure grid should not be None."
        assert v_grid is not None, "Velocity grid should not be None."
        assert a_grid is not None, "Alpha grid should not be None."
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
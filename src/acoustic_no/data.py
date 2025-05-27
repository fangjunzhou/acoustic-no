from torch.utils.data import Dataset
import numpy as np
import pathlib
import logging
import torch
from tqdm import tqdm
import json
import os
import tempfile
import shutil
from collections import OrderedDict


SAMPLE_PER_SCENE = 128


def preprocess_acoustic_dataset(
    data_dir: pathlib.Path,
    output_dir: pathlib.Path,
    num_chunks: int,
    depth: int = 8,
    compression_method: str = "diff",
    use_memmap: bool = True,
) -> None:
    """
    Preprocess acoustic dataset by loading checkpoints, sampling, and shuffling data.
    Saves the processed dataset to the specified output directory.
    
    Args:
        data_dir: Directory containing scene data
        output_dir: Directory to save preprocessed data
        num_chunks: Number of chunks to generate
        depth: Depth of the sequence for each sample
        compression_method: Method to compress data ('diff' or 'raw')
        use_memmap: Whether to use memory mapping for large arrays
    """
    from acoustic_no.data_utils import compress_sequence
    
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir = output_dir / "index"
    index_dir.mkdir(exist_ok=True)
    
    # Count number of scenes
    num_scenes = len([scene for scene in data_dir.iterdir() if scene.is_dir()])
    logger.info(f"Found {num_scenes} scenes in {data_dir}")
    
    # Dataset metadata
    metadata = {
        "depth": depth,
        "num_chunks": 0,
        "num_scenes": num_scenes,
        "total_samples": 0,
        "compression_method": compression_method,
        "use_memmap": use_memmap
    }
    
    for chunk in tqdm(range(num_chunks), desc="Processing chunks"):
        enough_samples = True
        np.random.seed(0)
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
                logger.info(f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}.")
                
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
                    chunk * SAMPLE_PER_SCENE : (chunk + 1) * SAMPLE_PER_SCENE
                ]
                
                # Ensure we have enough samples
                if len(indices) < SAMPLE_PER_SCENE:
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
            break
            
        # Concatenate all chunks for the current chunk
        chunk_p_grid = np.array(chunk_p_grid)
        chunk_v_grid = np.array(chunk_v_grid)
        chunk_a_grid = np.array(chunk_a_grid)
        
        # Shuffle the chunk data
        indices = np.random.permutation(len(chunk_p_grid))
        chunk_p_grid = chunk_p_grid[indices]
        chunk_v_grid = chunk_v_grid[indices]
        chunk_a_grid = chunk_a_grid[indices]
        
        # Create chunk directory
        chunk_dir = output_dir / f"chunk_{chunk}"
        chunk_dir.mkdir(exist_ok=True)
        
        # Apply compression to reduce file size if requested
        if compression_method == "diff":
            from acoustic_no.data_utils import compress_sequence
            # Compress each sequence to reduce storage space
            compressed_p = compress_sequence(chunk_p_grid, 'diff')
            compressed_v = compress_sequence(chunk_v_grid, 'diff')
            compressed_a = compress_sequence(chunk_a_grid, 'diff')
            
            # Save the compressed chunks
            p_chunk_file = chunk_dir / "pressure.npz"
            v_chunk_file = chunk_dir / "velocity.npz"
            a_chunk_file = chunk_dir / "alpha.npz"
            
            np.savez_compressed(p_chunk_file, **compressed_p)
            np.savez_compressed(v_chunk_file, **compressed_v)
            np.savez_compressed(a_chunk_file, **compressed_a)
            
            # Create index file for this chunk using relative paths (not absolute)
            index_file = index_dir / f"chunk_{chunk}_index.json"
            with open(index_file, 'w') as f:
                json.dump({
                    'num_samples': len(chunk_p_grid),
                    'files': {
                        'pressure': f"chunk_{chunk}/pressure.npz",
                        'velocity': f"chunk_{chunk}/velocity.npz",
                        'alpha': f"chunk_{chunk}/alpha.npz"
                    },
                    'compression': compression_method
                }, f)
        else:
            # Save in standard format for backward compatibility
            chunk_file = output_dir / f"chunk_{chunk}.npz"
            np.savez_compressed(
                chunk_file,
                pressure=chunk_p_grid,
                velocity=chunk_v_grid,
                alpha=chunk_a_grid,
            )
        
        # Update metadata
        metadata["num_chunks"] += 1
        metadata["total_samples"] += len(chunk_p_grid)
    
    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    logger.info(f"Preprocessed dataset saved to {output_dir}")
    logger.info(f"Total chunks: {metadata['num_chunks']}, total samples: {metadata['total_samples']}")


class ShuffledAcousticDataset(Dataset):
    def __init__(
        self,
        dataset_dir: pathlib.Path,
        cache_size: int = 2,
    ) -> None:
        """
        Load a preprocessed acoustic dataset from the specified directory.
        
        Args:
            dataset_dir: Directory containing the preprocessed dataset
            cache_size: Number of chunks to keep in memory cache (LRU policy)
        """
        self.logger = logging.getLogger(__name__)
        self.dataset_dir = dataset_dir
        self.index_dir = dataset_dir / "index"
        self.cache_size = cache_size
        self.chunk_cache = OrderedDict()  # LRU cache for chunks
        
        # Load metadata
        try:
            with open(dataset_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
            
            self.depth = metadata["depth"]
            self.num_chunks = metadata["num_chunks"]
            self.num_scenes = metadata["num_scenes"]
            self.length = metadata["total_samples"]
            self.compression_method = metadata.get("compression_method", "raw")
            self.use_memmap = metadata.get("use_memmap", False)
            
            # Set up indexing
            self.chunk_info = {}
            self.use_legacy_format = not self.index_dir.exists()
            
            if not self.use_legacy_format:
                # Load index files for each chunk
                for i in range(self.num_chunks):
                    index_file = self.index_dir / f"chunk_{i}_index.json"
                    if index_file.exists():
                        with open(index_file, 'r') as f:
                            self.chunk_info[i] = json.load(f)
            
            self.logger.info(f"Loaded dataset from {dataset_dir} with {self.length} samples")
            self.logger.info(f"Using {'compressed' if not self.use_legacy_format else 'legacy'} format")
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid dataset directory: {dataset_dir}. {str(e)}")

    def __len__(self):
        # Return the number of instances in the dataset.
        return self.length

    def _load_chunk_legacy(self, chunk_idx):
        """Load chunk data using the legacy format"""
        chunk_file = self.dataset_dir / f"chunk_{chunk_idx}.npz"
        data = np.load(chunk_file)
        p_grid = data["pressure"]
        v_grid = data["velocity"]
        a_grid = data["alpha"]
        return p_grid, v_grid, a_grid
    
    def _load_chunk_compressed(self, chunk_idx):
        """Load chunk data using the compressed format with optional memory mapping"""
        from acoustic_no.data_utils import decompress_sequence
        
        chunk_info = self.chunk_info[chunk_idx]
        
        # Properly resolve paths relative to dataset directory
        def resolve_path(file_path):
            # First try to load as is
            if os.path.exists(file_path):
                return file_path
            
            # Try as relative path from dataset directory
            filename = os.path.basename(file_path)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            
            # Build path relative to dataset dir
            relative_path = self.dataset_dir / parent_dir / filename
            if relative_path.exists():
                return relative_path
            
            # Try constructing from chunk and file type
            chunk_dir = self.dataset_dir / f"chunk_{chunk_idx}"
            if parent_dir.startswith("chunk_"):
                # For paths like 'chunk_0/pressure.npz'
                return chunk_dir / filename
            
            # If all else fails, return original and let it fail naturally
            return file_path
        
        # Resolve file paths
        p_path = resolve_path(chunk_info['files']['pressure'])
        v_path = resolve_path(chunk_info['files']['velocity'])
        a_path = resolve_path(chunk_info['files']['alpha'])
        
        # Load compressed data
        p_data = np.load(p_path)
        v_data = np.load(v_path)
        a_data = np.load(a_path)
        
        # Decompress if needed
        p_grid = decompress_sequence(dict(p_data))
        v_grid = decompress_sequence(dict(v_data))
        a_grid = decompress_sequence(dict(a_data))
        
        return p_grid, v_grid, a_grid
    
    def _get_chunk_data(self, chunk_idx):
        """Get chunk data from cache or load it"""
        # Check if the chunk is in cache
        if chunk_idx in self.chunk_cache:
            # Move to end of OrderedDict to mark as recently used
            self.chunk_cache.move_to_end(chunk_idx)
            return self.chunk_cache[chunk_idx]
        
        # Load the chunk data
        if self.use_legacy_format:
            chunk_data = self._load_chunk_legacy(chunk_idx)
        else:
            chunk_data = self._load_chunk_compressed(chunk_idx)
            
        # Add to cache
        self.chunk_cache[chunk_idx] = chunk_data
        
        # If cache is too large, remove least recently used item
        if len(self.chunk_cache) > self.cache_size:
            self.chunk_cache.popitem(last=False)
            
        return chunk_data
    
    def __getitem__(self, idx):
        # Calculate the chunk index and the index within the chunk.
        chunk_idx = idx // (SAMPLE_PER_SCENE * self.num_scenes)
        idx_in_chunk = idx % (SAMPLE_PER_SCENE * self.num_scenes)
        
        # Get chunk data from cache or load it
        p_grid, v_grid, a_grid = self._get_chunk_data(chunk_idx)
        
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

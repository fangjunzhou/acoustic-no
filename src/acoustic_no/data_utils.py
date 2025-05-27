import numpy as np
import pathlib
import logging
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

def compress_sequence(data: np.ndarray, compress_method: str = 'diff') -> Dict[str, Any]:
    """
    Compress a time sequence of data using different methods
    
    Args:
        data: Input data with shape (seq_len, ...)
        compress_method: Compression method ('diff' or 'raw')
        
    Returns:
        Dict containing compressed data and compression info
    """
    if compress_method == 'raw':
        return {'data': data, 'method': 'raw'}
    
    elif compress_method == 'diff':
        # Store first frame and differences
        first_frame = data[0].copy()
        diffs = np.diff(data, axis=0)
        
        return {
            'first_frame': first_frame,
            'diffs': diffs,
            'method': 'diff',
            'original_shape': data.shape
        }
    
    else:
        logger.warning(f"Unknown compression method: {compress_method}. Using raw storage.")
        return {'data': data, 'method': 'raw'}

def decompress_sequence(compressed_data: Dict[str, Any]) -> np.ndarray:
    """
    Decompress data compressed by compress_sequence
    
    Args:
        compressed_data: Dict containing compressed data
        
    Returns:
        Original decompressed data
    """
    method = compressed_data.get('method')
    
    if method == 'raw':
        return compressed_data['data']
    
    elif method == 'diff':
        first_frame = compressed_data['first_frame']
        diffs = compressed_data['diffs']
        original_shape = compressed_data.get('original_shape')
        
        # Reconstruct from first frame and differences
        result = np.zeros(original_shape, dtype=first_frame.dtype)
        result[0] = first_frame
        
        for i in range(1, original_shape[0]):
            result[i] = result[i-1] + diffs[i-1]
            
        return result
    
    else:
        logger.error(f"Unknown decompression method: {method}")
        raise ValueError(f"Unknown decompression method: {method}")

def create_index_file(chunk_dir: pathlib.Path, samples_per_chunk: int) -> Dict[str, Any]:
    """
    Create an index file mapping sample indices to file positions
    
    Args:
        chunk_dir: Directory containing chunk files
        samples_per_chunk: Number of samples per chunk
        
    Returns:
        Dictionary with indexing information
    """
    index_data = {}
    chunk_files = sorted(list(chunk_dir.glob("chunk_*.npz")))
    
    for i, chunk_file in enumerate(chunk_files):
        chunk_idx = int(chunk_file.stem.split('_')[-1])
        start_idx = chunk_idx * samples_per_chunk
        end_idx = start_idx + samples_per_chunk
        
        index_data[chunk_idx] = {
            'file': str(chunk_file),
            'start_idx': start_idx,
            'end_idx': end_idx,
        }
    
    return index_data

#!/usr/bin/env python
# filepath: /Users/fangjun/Documents/stanford/acoustic-no/scripts/convert-dataset-format.py
import pathlib
import argparse
import logging
import numpy as np
import json
import os
import shutil
from tqdm import tqdm

from acoustic_no.data_utils import compress_sequence

def convert_dataset(input_dir: pathlib.Path, output_dir: pathlib.Path, compression_method: str = "diff"):
    """
    Convert an existing dataset to the new compressed format
    
    Args:
        input_dir: Directory containing the old format dataset
        output_dir: Directory to save the converted dataset
        compression_method: Method to compress data ('diff' or 'raw')
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir = output_dir / "index"
    index_dir.mkdir(exist_ok=True)
    
    # Copy metadata file
    with open(input_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Update metadata
    metadata["compression_method"] = compression_method
    metadata["use_memmap"] = True
    
    # Save updated metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    # Process each chunk
    num_chunks = metadata["num_chunks"]
    for chunk_idx in tqdm(range(num_chunks), desc="Converting chunks"):
        # Load chunk file
        chunk_file = input_dir / f"chunk_{chunk_idx}.npz"
        data = np.load(chunk_file)
        p_grid = data["pressure"]
        v_grid = data["velocity"]
        a_grid = data["alpha"]
        
        # Create chunk directory
        chunk_dir = output_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(exist_ok=True)
        
        # Apply compression
        compressed_p = compress_sequence(p_grid, compression_method)
        compressed_v = compress_sequence(v_grid, compression_method)
        compressed_a = compress_sequence(a_grid, compression_method)
        
        # Save the compressed chunks
        p_chunk_file = chunk_dir / "pressure.npz"
        v_chunk_file = chunk_dir / "velocity.npz"
        a_chunk_file = chunk_dir / "alpha.npz"
        
        np.savez_compressed(p_chunk_file, **compressed_p)
        np.savez_compressed(v_chunk_file, **compressed_v)
        np.savez_compressed(a_chunk_file, **compressed_a)
        
        # Create index file for this chunk
        index_file = index_dir / f"chunk_{chunk_idx}_index.json"
        with open(index_file, 'w') as f:
            json.dump({
                'num_samples': len(p_grid),
                'files': {
                    'pressure': str(p_chunk_file),
                    'velocity': str(v_chunk_file),
                    'alpha': str(a_chunk_file)
                },
                'compression': compression_method
            }, f)
    
    logger.info(f"Converted {num_chunks} chunks from {input_dir} to {output_dir}")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Convert dataset from old format to new compressed format")
    parser.add_argument("--input-dir", type=pathlib.Path, required=True,
                       help="Directory containing the old format dataset")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True,
                       help="Directory to save the converted dataset")
    parser.add_argument("--compression", type=str, default="diff", choices=["diff", "raw"],
                       help="Compression method to use")
    
    args = parser.parse_args()
    
    # Convert the dataset
    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        compression_method=args.compression
    )

if __name__ == "__main__":
    main()

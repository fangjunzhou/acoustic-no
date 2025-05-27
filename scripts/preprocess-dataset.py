#!/usr/bin/env python
# filepath: /Users/fangjun/Documents/stanford/acoustic-no/scripts/preprocess-dataset.py
import pathlib
import argparse
import logging

from acoustic_no.data import preprocess_acoustic_dataset

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Preprocess acoustic dataset for training")
    parser.add_argument("--data-dir", type=pathlib.Path, required=True,
                       help="Directory containing the raw dataset with scene folders")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True,
                       help="Directory to save the preprocessed dataset")
    parser.add_argument("--num-chunks", type=int, default=16,
                       help="Number of chunks to generate")
    parser.add_argument("--depth", type=int, default=8,
                       help="Depth of the sequence for each sample")
    
    args = parser.parse_args()
    
    # Preprocess the dataset
    preprocess_acoustic_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_chunks=args.num_chunks,
        depth=args.depth
    )

if __name__ == "__main__":
    main()

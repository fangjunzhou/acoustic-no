#!/usr/bin/env python
# filepath: /home/ubuntu/acoustic-no/scripts/train-cno.py
"""
Training script for Continuous Neural Operator (CNO) model on acoustic wave propagation data.
Based on the notebook acoustic-cno.ipynb.
"""

import pathlib
import torch
import torch.nn as nn
from torch.utils.data import Subset, random_split
import argparse
from tqdm import tqdm
import logging
import numpy as np

from acoustic_no.cno.cno_model import CNOModel
from acoustic_no.data import AcousticDataset, ShuffledAcousticDataset, preprocess_acoustic_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fraction of data to use for training
FRACTION_TRAIN = 0.8

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a CNO model on acoustic wave data")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("resources/dataset/training"),
                       help="Directory containing the raw dataset")
    parser.add_argument("--processed-dir", type=pathlib.Path, default=pathlib.Path("resources/dataset/processed"),
                       help="Directory to save/load the processed dataset")
    parser.add_argument("--preprocess", action="store_true", 
                       help="Preprocess the dataset before training")
    parser.add_argument("--use-preprocess", action="store_true", 
                      help="Preprocess the dataset before training")
    parser.add_argument("--num-chunks", type=int, default=16,
                       help="Number of chunks to use for preprocessing")
    parser.add_argument("--dataset-size", type=int, default=-1,
                        help="Size of the dataset to use for training")
    parser.add_argument("--depth", type=int, default=8,
                         help="Depth of the sequence for each sample")
    parser.add_argument("--num-epochs", type=int, default=32,
                         help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32,
                         help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                         help="Learning rate for optimizer")
    parser.add_argument("--hidden-channels", type=int, nargs='+', default=[128, 128, 256],
                         help="Number of hidden channels in each layer")
    parser.add_argument("--layer-sizes", type=int, nargs='+', default=[3, 3, 4],
                         help="Number of residual blocks in each layer")
    parser.add_argument("--num-workers", type=int, default=4,
                         help="Number of workers for data loading")
    parser.add_argument("--save-dir", type=pathlib.Path, default=pathlib.Path("resources/models/cno"),
                         help="Directory to save the trained model")
    return parser.parse_args()

def train(model, train_loader, val_loader, optimizer, criterion, device, args):
    """Train the CNO model."""
    best_val_loss = float('inf')
    save_path = args.save_dir / "best_model.pth"
    args.save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch") as t:
            for i, batch in enumerate(t):
                inputs, targets = batch["x"], batch["y"]
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                t.set_postfix(loss=running_loss/(i+1))
        
        train_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch["x"], batch["y"]
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Validation Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"Saving best model with validation loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), save_path)

def main():
    """Main function to run the training script."""
    args = parse_args()
    
    # Set the device for training
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Preprocess data if requested
    if args.preprocess:
        logger.info(f"Preprocessing dataset from {args.data_dir} to {args.processed_dir}...")
        preprocess_acoustic_dataset(
            data_dir=args.data_dir,
            output_dir=args.processed_dir,
            num_chunks=args.num_chunks,
            depth=args.depth
        )
    
    # Load the dataset
    if args.use_preprocess:
        dataset = ShuffledAcousticDataset(
            dataset_dir=args.processed_dir,
        )
    else:
        dataset = AcousticDataset(
            data_dir=args.data_dir,
            depth=args.depth,
        )
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Dataset depth: {dataset.depth}")
    
    # Verify dataset depth
    depth = dataset.depth
    if depth != args.depth:
        raise ValueError(f"Expected dataset depth {args.depth}, but got {depth}.")
    

    # Limit dataset size if specified
    if args.dataset_size > 0 and args.dataset_size < len(dataset):
        logger.info(f"Limiting dataset size to {args.dataset_size} samples")
        indx = np.random.choice(
            len(dataset), 
            size=args.dataset_size, 
            replace=False
        )
        dataset = Subset(dataset, indx)
    else:
        logger.info(f"Using full dataset size: {len(dataset)} samples")
    logger.info(f"Dataset size: {len(dataset)}")

    # Split the dataset into training and validation sets
    if not args.use_preprocess:
        train_size = int(FRACTION_TRAIN * len(dataset))
        train_dataset = Subset(dataset, range(train_size))
        val_dataset = Subset(dataset, range(train_size, len(dataset)))
    else:
        train_dataset, val_dataset = random_split(
            dataset,
            [int(FRACTION_TRAIN * len(dataset)), len(dataset) - int(FRACTION_TRAIN * len(dataset))],
            generator=torch.Generator().manual_seed(42)
        )
    logger.info(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.use_preprocess,  # Shuffle only if not using preprocessed data
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=not args.use_preprocess,  # Shuffle only if not using preprocessed data
        num_workers=args.num_workers,
    )
    
    # Ensure hidden_channels and layer_sizes have the same length
    if len(args.hidden_channels) != len(args.layer_sizes):
        raise ValueError("hidden_channels and layer_sizes must have the same length")
    
    # Initialize the CNO model
    model = CNOModel(
        input_channels=depth * 3 + 1,
        hidden_channels=args.hidden_channels,
        layer_sizes=args.layer_sizes,
        output_channels=depth
    )
    model.to(device)
    logger.info(f"Initialized model: {model}")
    
    # Initialize training components
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    train(model, train_loader, val_loader, optimizer, criterion, device, args)
    
    logger.info(f"Training complete. Best model saved to {args.save_dir / 'best_model.pth'}")

if __name__ == "__main__":
    main()

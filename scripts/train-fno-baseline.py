#!/usr/bin/env python
# filepath: /home/ubuntu/acoustic-no/scripts/train-fno-baseline.py
"""
Training script for Fourier Neural Operator (FNO) model on acoustic wave propagation data.
"""

import pathlib
import torch
from torch.utils.data import Subset, random_split
from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss
from neuralop.training.incremental import IncrementalFNOTrainer
from neuralop.data.transforms.data_processors import IncrementalDataProcessor
import logging
import argparse
import numpy as np

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
    parser = argparse.ArgumentParser(description="Train an FNO baseline model")
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
    parser.add_argument("--depth", type=int, default=8,
                      help="Depth of the sequence for each sample")
    parser.add_argument("--dataset-size", type=int, default=-1,
                        help="Size of the dataset to use for training")
    parser.add_argument("--num-epochs", type=int, default=32,
                      help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64,
                      help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=10,
                      help="Number of workers for data loading")
    parser.add_argument("--learning-rate", type=float, default=8e-3,
                      help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                      help="Weight decay for optimizer")
    parser.add_argument("--n-modes", type=int, nargs='+', default=[16, 16],
                      help="Number of Fourier modes to use")
    parser.add_argument("--hidden-channels", type=int, default=128,
                      help="Number of hidden channels")
    parser.add_argument("--n-layers", type=int, default=8,
                      help="Number of layers in the FNO model")
    parser.add_argument("--save-dir", type=pathlib.Path, default=pathlib.Path("resources/models/fno_baseline"),
                      help="Directory to save the trained model")
    return parser.parse_args()

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
    logger.info(f"Loaded dataset from {args.processed_dir} with {len(dataset)} samples")
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
    val_loader = {
        "64x64": torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=not args.use_preprocess,  # Shuffle only if not using preprocessed data
            num_workers=args.num_workers,
        )
    }
    
    # Initialize the FNO model
    model = FNO(
        n_modes=tuple(args.n_modes),
        in_channels=depth * 3 + 1,
        out_channels=depth,
        n_layers=args.n_layers,
        hidden_channels=args.hidden_channels,
        projection_channel_ratio=2,
    )
    model.to(device)
    logger.info(f"Initialized FNO model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Setup data processors and loss functions
    data_transform = IncrementalDataProcessor(
        in_normalizer=None,
        out_normalizer=None,
        device=device,
        subsampling_rates=[8, 4, 2, 1],
        dataset_resolution=64,
        dataset_indices=[2, 3],
        epoch_gap=8,
        verbose=True,
    )
    data_transform = data_transform.to(device)
    
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = h1loss
    eval_losses = {"h1": h1loss, "l2": l2loss}
    
    # Ensure the save directory exists
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the trainer
    trainer = IncrementalFNOTrainer(
        model=model,
        n_epochs=args.num_epochs,
        data_processor=data_transform,
        device=device,
        verbose=True,
        incremental_loss_gap=False,
        incremental_grad=True,
        incremental_grad_eps=0.9999,
        incremental_loss_eps=0.001,
        incremental_buffer=5,
        incremental_max_iter=1,
        incremental_grad_max_iter=2,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train(
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
        save_best="64x64_h1",
        save_dir=args.save_dir
    )
    
    logger.info(f"Training complete. Best model saved to {args.save_dir}")

if __name__ == "__main__":
    main()

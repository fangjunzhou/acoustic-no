import pathlib
import torch
from torch.utils.data import Subset, random_split
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss
from neuralop.training.incremental import IncrementalFNOTrainer
from neuralop.data.transforms.data_processors import IncrementalDataProcessor
from tqdm import tqdm

from acoustic_no.data import ShuffledAcousticDataset

# Prediction depth.
DEPTH = 64
# Number of training and validation samples.
FRACTION_TRAIN = 0.8

# Use the GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load the dataset
dataset = ShuffledAcousticDataset(
    data_dir=pathlib.Path("resources/dataset/training"),
    num_chunks=16,
    depth=DEPTH,
)
print(f"Dataset size: {len(dataset)}")

# Split the dataset into training and validation sets
train_dataset = Subset(
    dataset,
    range(int(FRACTION_TRAIN * len(dataset))),
)
val_dataset = Subset(
    dataset,
    range(int(FRACTION_TRAIN * len(dataset)), len(dataset)),
)
# Create a data loader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)
val_loader = {
    "64x64": torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
}

model = FNO(
    n_modes=(16, 16),
    in_channels=DEPTH * 3 + 1,
    out_channels=DEPTH,
    hidden_channels=64,
    projection_channel_ratio=2,
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

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

# Finally pass all of these to the Trainer
trainer = IncrementalFNOTrainer(
    model=model,
    n_epochs=32,
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

trainer.train(
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
    save_best="64x64_h1",
)

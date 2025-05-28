import pathlib
import torch
from torch.utils.data import Subset, random_split
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss
import time
from tqdm import tqdm

from acoustic_no.data import AcousticDataset
from acoustic_no.models import UFNO

print("\n=== U-FNO Training Script ===\n")
print("Initializing training configuration...")

# Prediction depth
DEPTH = 64
# Number of training and validation samples
N_TRAIN, N_VAL = 1024, 16
# Number of epochs
N_EPOCHS = 32

print(f"Configuration:")
print(f"- Prediction depth: {DEPTH}")
print(f"- Training samples: {N_TRAIN}")
print(f"- Validation samples: {N_VAL}")
print(f"- Number of epochs: {N_EPOCHS}")

# Use the GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

print("\nLoading dataset...")
# Load the dataset
dataset = AcousticDataset(
    data_dir=pathlib.Path("resources/small_dataset/training"),
    depth=DEPTH,
)
print(f"Total dataset size: {len(dataset)}")

print("\nSplitting dataset...")
# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"- Full training set: {len(train_dataset)}")
print(f"- Full validation set: {len(val_dataset)}")

# Use a random subset of the dataset for training
idx_train = torch.randperm(len(train_dataset))[:N_TRAIN]
train_dataset = Subset(train_dataset, idx_train)
idx_val = torch.randperm(len(val_dataset))[:N_VAL]
val_dataset = Subset(val_dataset, idx_val)
print(f"- Using {len(train_dataset)} training samples")
print(f"- Using {len(val_dataset)} validation samples")

print("\nCreating data loaders...")
# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)
print("Data loaders created")

print("\nInitializing model...")
# Initialize the U-FNO model
model = UFNO(
    n_modes=(16, 16),
    in_channels=DEPTH * 3 + 1,  # pressure, velocity (x,y), and alpha
    out_channels=DEPTH,  # pressure prediction
    width=64,
    n_layers=6
)
model.to(device)
print("Model initialized")

print("\nSetting up training components...")
# Setup optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
print("Optimizer and scheduler configured")

# Setup loss functions
print("Setting up loss functions...")
train_loss = H1Loss(d=2)
eval_losses = {
    "h1": H1Loss(d=2),
    "l2": LpLoss(d=2, p=2)
}
print("Loss functions configured")

print("\n=== Starting Training ===")
start_time = time.time()

best_val_loss = float('inf')
for epoch in range(N_EPOCHS):
    # Training
    model.train()
    train_losses = []
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move data to device
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        
        # Forward pass
        y_pred = model(x)
        loss = train_loss(y_pred, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        progress_bar.set_postfix({'loss': sum(train_losses) / len(train_losses)})
    
    # Validation
    model.eval()
    val_losses = {name: [] for name in eval_losses.keys()}
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            y_pred = model(x)
            
            for name, criterion in eval_losses.items():
                loss = criterion(y_pred, y)
                val_losses[name].append(loss.item())
    
    # Print epoch results
    train_loss_avg = sum(train_losses) / len(train_losses)
    val_loss_str = ", ".join([f"{name}: {sum(losses)/len(losses):.4f}" 
                             for name, losses in val_losses.items()])
    print(f"Epoch {epoch+1}/{N_EPOCHS}")
    print(f"Train Loss: {train_loss_avg:.4f}")
    print(f"Val Losses: {val_loss_str}")
    
    # Save best model
    val_h1_loss = sum(val_losses['h1']) / len(val_losses['h1'])
    if val_h1_loss < best_val_loss:
        best_val_loss = val_h1_loss
        torch.save(model.state_dict(), "ckpt/best_model_state_dict_64x64_h1.pt")
    
    # Update learning rate
    scheduler.step()

end_time = time.time()
training_duration = end_time - start_time
hours = int(training_duration // 3600)
minutes = int((training_duration % 3600) // 60)
seconds = int(training_duration % 60)

print(f"\n=== Training Complete ===")
print(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
print(f"Best model saved as: ckpt/best_model_state_dict_64x64_h1.pt") 
import pathlib
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from tqdm import tqdm
import time

from acoustic_no.data import AcousticDataset
from acoustic_no.models import UFNO
from neuralop import LpLoss, H1Loss

print("\n=== U-FNO Evaluation Script ===\n")

# Prediction depth
DEPTH = 64
print(f"Configuration:")
print(f"- Prediction depth: {DEPTH}")

# Use the GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

print("\nLoading test dataset...")
# Load the test dataset
dataset = AcousticDataset(
    data_dir=pathlib.Path("resources/small_dataset/testing"),
    depth=DEPTH,
)
print(f"Test dataset size: {len(dataset)}")

print("\nCreating data loader...")
# Create data loader
test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)
print("Data loader created")

print("\nLoading trained model...")
# Load the trained model
model = UFNO(
    n_modes=(16, 16),
    in_channels=DEPTH * 3 + 1,
    out_channels=DEPTH,
    width=64,
    n_layers=6,
)
model.load_state_dict(torch.load("ckpt/best_model_state_dict_64x64_h1.pt", map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully")

print("\nSetting up evaluation metrics...")
# Setup loss functions
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

# Evaluation metrics
total_l2_loss = 0.0
total_h1_loss = 0.0
num_samples = 0

# Lists to store sample predictions for visualization
sample_inputs = []
sample_preds = []
sample_targets = []
print("Metrics initialized")

print("\n=== Starting Evaluation ===")
start_time = time.time()

print("\nProcessing test batches...")
with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
        # Forward pass
        pred = model(x)
        
        # Calculate losses
        l2 = l2loss(pred, y)
        h1 = h1loss(pred, y)
        
        total_l2_loss += l2.item() * x.size(0)
        total_h1_loss += h1.item() * x.size(0)
        num_samples += x.size(0)
        
        # Store first batch for visualization
        if len(sample_inputs) == 0:
            print(f"\nStoring sample batch {batch_idx} for visualization...")
            sample_inputs.append(x[0].cpu().numpy())
            sample_preds.append(pred[0].cpu().numpy())
            sample_targets.append(y[0].cpu().numpy())

# Calculate average losses
avg_l2_loss = total_l2_loss / num_samples
avg_h1_loss = total_h1_loss / num_samples

print("\n=== Evaluation Results ===")
print(f"Average L2 Loss: {avg_l2_loss:.6f}")
print(f"Average H1 Loss: {avg_h1_loss:.6f}")

print("\n=== Generating Visualizations ===")
# Create output directory for plots
output_dir = pathlib.Path("outputs")
output_dir.mkdir(exist_ok=True)
print("Created output directory")

print("\n1. Plotting initial conditions...")
# Plot initial conditions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].imshow(sample_inputs[0][0])  # Initial pressure
plt.colorbar(im0, ax=axes[0])
axes[0].set_title("Initial Pressure")

im1 = axes[1].imshow(sample_inputs[0][1:3].mean(axis=0))  # Average velocity
plt.colorbar(im1, ax=axes[1])
axes[1].set_title("Initial Velocity (avg)")

im2 = axes[2].imshow(sample_inputs[0][3:].mean(axis=0))  # Alpha field
plt.colorbar(im2, ax=axes[2])
axes[2].set_title("Alpha Field (avg)")

plt.tight_layout()
plt.savefig(output_dir / "initial_conditions.png")
plt.close()
print("Initial conditions plot saved")

print("\n2. Creating prediction animation...")
# Create animation of predictions vs ground truth
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Pressure Field Evolution")

pred_data = sample_preds[0]
target_data = sample_targets[0]

vmin = min(pred_data.min(), target_data.min())
vmax = max(pred_data.max(), target_data.max())

im1 = ax1.imshow(pred_data[0], vmin=vmin, vmax=vmax)
im2 = ax2.imshow(target_data[0], vmin=vmin, vmax=vmax)
plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)

ax1.set_title("Prediction")
ax2.set_title("Ground Truth")

def update(frame):
    im1.set_array(pred_data[frame])
    im2.set_array(target_data[frame])
    return [im1, im2]

anim = FuncAnimation(fig, update, frames=DEPTH, interval=50, blit=True)
writer = PillowWriter(fps=10)
anim.save(output_dir / "pressure_evolution.gif", writer=writer)
plt.close()
print("Animation saved")

print("\n3. Plotting error evolution...")
# Plot error evolution
error_evolution = np.abs(pred_data - target_data).mean(axis=(1, 2))
plt.figure(figsize=(10, 5))
plt.plot(error_evolution, 'b-', label='Mean Absolute Error')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.title('Error Evolution Over Time')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / "error_evolution.png")
plt.close()
print("Error evolution plot saved")

print("\n4. Plotting error distribution...")
# Plot error distribution
plt.figure(figsize=(10, 5))
sns.histplot(error_evolution, bins=30, kde=True)
plt.xlabel('Error Magnitude')
plt.ylabel('Count')
plt.title('Distribution of Prediction Errors')
plt.savefig(output_dir / "error_distribution.png")
plt.close()
print("Error distribution plot saved")

end_time = time.time()
eval_duration = end_time - start_time
hours = int(eval_duration // 3600)
minutes = int((eval_duration % 3600) // 60)
seconds = int(eval_duration % 60)

print("\n=== Evaluation Complete ===")
print(f"Total evaluation time: {hours:02d}:{minutes:02d}:{seconds:02d}")
print(f"All results saved in outputs/ directory") 
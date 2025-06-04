import torch
import logging
import pathlib
import argparse
from tqdm.auto import tqdm
from neuralop import LpLoss, H1Loss
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
import numpy as np

from acoustic_no.cno.cno_model import CNOModel
from acoustic_no.data import AcousticDataset

logger = logging.getLogger(__name__)

# Create output directory for plots
output_dir = pathlib.Path("outputs/cno")
output_dir.mkdir(exist_ok=True)
print("Created output directory")

# Set up argument parser
parser = argparse.ArgumentParser(description="Evaluate CNO model on a dataset.")
# Set up logging
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level.",
)
# Model state dictionary path
parser.add_argument(
    "--model-path",
    type=str,
    default="resources/models/cno/best_model.pth",
    help="Path to the model state dictionary.",
)
# Dataset path
parser.add_argument(
    "--dataset-path",
    type=str,
    default="resources/dataset/testing",
    help="Path to the dataset directory.",
)
args = parser.parse_args()

# Use the GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger.info(f"Using device: {device}")

# Load the dataset
dataset = AcousticDataset(data_dir=pathlib.Path(args.dataset_path), depth=64)
depth = dataset.depth
dataset = Subset(dataset, range(0, len(dataset), 32))  # Subsample for faster evaluation
logger.info(f"Dataset loaded with {len(dataset)} samples.")

# Load the CNO model
model = CNOModel(
    input_channels=depth * 3 + 1,
    hidden_channels=[128, 128, 256],
    layer_sizes=[3, 3, 4],
    output_channels=depth,
).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))

# Display a sample from the dataset
data = dataset[len(dataset) // 2]
p = data["y"]
v = data["v"]
a = data["a"]
# Move velocity x and y to r and g channels
v = v.reshape(-1, 2, v.shape[1], v.shape[2])
v = v.permute(0, 2, 3, 1)
# Extend the velocity to 3 channels
v = torch.cat([v, torch.zeros_like(v[:, :, :, 0:1])], dim=3)
# Normalize the velocity
v = (v - v.min()) / (v.max() - v.min())
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(p[0, :, :], interpolation="nearest")
ax[0].set_title("Pressure")
ax[1].imshow(v[0, :, :], interpolation="nearest")
ax[1].set_title("Velocity")
ax[2].imshow(a[0, :, :], interpolation="nearest")
ax[2].set_title("Alpha")
# Save the figure
plt.savefig("outputs/cno/initial_conditions.png")

# Inference
model.eval()
x = data["x"]
p = data["y"]
v = data["v"]
a = data["a"]
with torch.no_grad():
    pred = model(x.unsqueeze(0).to(device))

# Plot the results
fig, ax = plt.subplots(1, 3, figsize=(12, 6))
ax[0].imshow(pred[0, -1].cpu().numpy(), cmap="viridis", vmin=-10, vmax=10)
ax[0].set_title("Predicted Pressure")
ax[1].imshow(p[-1].cpu().numpy(), cmap="viridis", vmin=-10, vmax=10)
ax[1].set_title("Ground Truth Pressure")
ax[2].imshow(
    pred[0, -1].cpu().numpy() - p[-1].cpu().numpy(), cmap="viridis", vmin=-10, vmax=10
)
ax[2].set_title("Difference")
plt.tight_layout()
plt.savefig("outputs/cno/inference_results.png")

print("Creating prediction animation...")
# Create animation of predictions vs ground truth
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Pressure Field Evolution")

pred_data = pred[-1]
target_data = p

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


anim = FuncAnimation(fig, update, frames=depth, interval=50, blit=True)
writer = PillowWriter(fps=10)
anim.save(output_dir / "pressure_evolution.gif", writer=writer)
plt.close()
print("Animation saved")

print("5. Plotting relative error map...")
# Add relative error visualization
fig, ax = plt.subplots(figsize=(10, 5))
rel_error_map = np.abs(pred_data - target_data) / (np.abs(target_data) + 1e-8)
im = ax.imshow(rel_error_map.mean(axis=0))
plt.colorbar(im, ax=ax)
ax.set_title("Mean Relative Error Map")
plt.savefig(output_dir / "relative_error_map.png")
plt.close()
print("Relative error map saved")

# Evaluate the model on the test dataset
model.eval()
# Setup loss functions
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
mse_criterion = torch.nn.MSELoss()

# Evaluation metrics
total_l2_loss = 0.0
total_h1_loss = 0.0
total_mse_loss = 0.0
total_rel_l2_error = 0.0
total_max_error = 0.0
num_samples = 0


error_evolution = np.zeros(64)

with torch.no_grad():
    for i in tqdm(range(len(dataset)), desc="Evaluating", unit="sample"):
        data = dataset[i]
        x = data["x"].unsqueeze(0).to(device)
        y = data["y"].unsqueeze(0).to(device)
        pred = model(x)

        # Calculate losses
        l2 = l2loss(pred, y)
        h1 = h1loss(pred, y)
        mse = mse_criterion(pred, y)

        # Calculate relative L2 error
        rel_l2 = torch.norm(pred - y, p=2) / torch.norm(y, p=2)

        # Calculate maximum error
        max_error = torch.max(torch.abs(pred - y))

        # Accumulate metrics
        total_l2_loss += l2.item() * x.size(0)
        total_h1_loss += h1.item() * x.size(0)
        total_mse_loss += mse.item() * x.size(0)
        total_rel_l2_error += rel_l2.item() * x.size(0)
        total_max_error += max_error.item() * x.size(0)
        num_samples += 1
        # Store error evolution
        error_evolution += torch.abs(pred - y).mean(dim=(1, 2)).cpu().numpy()[0]

# Normalize error evolution
error_evolution /= num_samples

print("Plotting error evolution...")
# Plot error evolution
plt.figure(figsize=(10, 5))
plt.plot(error_evolution, "b-", label="Mean Absolute Error")
plt.xlabel("Time Step")
plt.ylabel("Error")
plt.title("CNO Error Evolution Over Time")
plt.legend()
plt.grid(True)
plt.savefig(output_dir / "error_evolution.png")
plt.close()
print("Error evolution plot saved")

print("Plotting error distribution...")
# Plot error distribution
plt.figure(figsize=(10, 5))
sns.histplot(error_evolution, bins=30, kde=True)
plt.xlabel("Error Magnitude")
plt.ylabel("Count")
plt.title("Distribution of Prediction Errors")
plt.savefig(output_dir / "error_distribution.png")
plt.close()
print("Error distribution plot saved")

# Calculate average metrics
avg_l2_loss = total_l2_loss / num_samples
avg_h1_loss = total_h1_loss / num_samples
avg_mse_loss = total_mse_loss / num_samples
avg_rel_l2_error = total_rel_l2_error / num_samples
avg_max_error = total_max_error / num_samples

print("\n=== Evaluation Results ===")
print(f"Average L2 Loss: {avg_l2_loss:.6f}")
print(f"Average H1 Loss: {avg_h1_loss:.6f}")
print(f"Average MSE Loss: {avg_mse_loss:.6f}")
print(f"Average Relative L2 Error: {avg_rel_l2_error:.6f}")
print(f"Average Maximum Error: {avg_max_error:.6f}")

# Save metrics to file
metrics_file = output_dir / "evaluation_metrics.txt"
with open(metrics_file, "w") as f:
    f.write("=== CNO Evaluation Results ===\n")
    f.write(f"Test Dataset Size: {len(dataset)}\n")
    f.write(f"Average L2 Loss: {avg_l2_loss:.6f}\n")
    f.write(f"Average H1 Loss: {avg_h1_loss:.6f}\n")
    f.write(f"Average MSE Loss: {avg_mse_loss:.6f}\n")
    f.write(f"Average Relative L2 Error: {avg_rel_l2_error:.6f}\n")
    f.write(f"Average Maximum Error: {avg_max_error:.6f}\n")

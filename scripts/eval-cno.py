import torch
import logging
import pathlib
import argparse
from tqdm.auto import tqdm
from neuralop import LpLoss, H1Loss

from acoustic_no.cno.cno_model import CNOModel
from acoustic_no.data import AcousticDataset

logger = logging.getLogger(__name__)

# Set up argument parser
parser = argparse.ArgumentParser(description="Evaluate CNO model on a dataset.")
# Set up logging
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level."
)
# Model state dictionary path
parser.add_argument(
    "--model-path",
    type=str,
    default="resources/models/cno/best_model.pth",
    help="Path to the model state dictionary."
)
# Dataset path
parser.add_argument(
    "--dataset-path",
    type=str,
    default="resources/dataset/testing",
    help="Path to the dataset directory."
)
args = parser.parse_args()

# Use the GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger.info(f"Using device: {device}")

# Load the dataset
dataset = AcousticDataset(
    data_dir=pathlib.Path(args.dataset_path),
    depth=64
)
depth = dataset.depth
logger.info(f"Dataset loaded with {len(dataset)} samples.")

# Load the CNO model
model = CNOModel(
    input_channels=depth * 3 + 1,
    hidden_channels=[128, 128, 256],
    layer_sizes=[3, 3, 4],
    output_channels=depth
).to(device)
model.load_state_dict(
    torch.load(
        args.model_path,
        map_location=device
    )
)

# Evaluate the model on the test dataset
model.eval()
criteria = {
    "mse": torch.nn.MSELoss(),
    "l2": LpLoss(d=2, p=2),
    "h1": H1Loss(d=2)
}
total_loss = {key: 0.0 for key in criteria.keys()}

with torch.no_grad():
    for i in tqdm(range(len(dataset)), desc="Evaluating", unit="sample"):
        data = dataset[i]
        x = data["x"].unsqueeze(0).to(device)
        y_true = data["y"].unsqueeze(0).to(device)
        y_pred = model(x)

        for key in criteria.keys():
            loss = criteria[key](y_pred, y_true)
            total_loss[key] += loss.item()

average_loss = {key: total / len(dataset) for key, total in total_loss.items()}
print(f"Average Loss: {average_loss}")
# Code pulled from eval.ipynb
# This script is specifically for generating a full animation using iterative sampling

from neuralop.models import FNO
import torch
import pathlib


from acoustic_no.data import AcousticDataset
from acoustic_no.cno.cno_model import CNOModel 
import acoustic_no.utils.eval as eval_utils

# Use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
dataset = AcousticDataset(
    data_dir=pathlib.Path("resources/small_dataset/testing"),
    depth=64
)
print(f"Dataset size: {len(dataset)}")

depth = dataset.depth
print(f"Dataset depth: {depth}")
'''
fno = FNO(
    n_modes=(16, 16),
    in_channels=depth * 3 + 1,
    out_channels=depth,
    n_layers=8,
    hidden_channels=128,
    projection_channel_ratio=2,
)
fno.load_state_dict(
    torch.load(
        "resources/models/fno_baseline/best_model_state_dict.pt",
        map_location=device,
        weights_only=False
    )
)
fno.to(device)

tfno = FNO(
    n_modes=(16, 16),
    in_channels=depth * 3 + 1,
    out_channels=depth,
    n_layers=8,
    hidden_channels=128,
    projection_channel_ratio=2,
    factorization="Tucker"
)
tfno.load_state_dict(
    torch.load(
        "resources/models/tfno/best_model_state_dict.pt",
        map_location=device,
        weights_only=False
    )
)
tfno.to(device)
'''
cno = CNOModel(
    input_channels=depth * 3 + 1,
    hidden_channels=[128, 128, 256],
    layer_sizes=[3, 3, 4],
    output_channels=depth
)
cno.load_state_dict(
    torch.load(
        "resources/models/cno/best_model.pth",
        map_location=device
    )
)
cno.to(device)

# Warning: It will probably take >30 minutes to generate a full animation
eval_utils.sample_iterative(cno, dataset, device, 
                            kind="animation", name="CNO", index=1000,
                            path="outputs/cno_iterative.gif")

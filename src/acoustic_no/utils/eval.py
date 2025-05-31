import torch
from tqdm import tqdm
from neuralop import LpLoss, H1Loss

def evaluate_models(models, dataset, device=None, print_results=True):
    """
    Evaluate multiple models on a dataset and return the results.

    Parameters:
    - models (dict): A dictionary where keys are model names and values are the
    model instances.
    - dataset (AcousticDataset): The dataset to evaluate the models on.
    - print_results (bool): Whether to print the results (default True).

    Returns:
    - results (dict): A dictionary where keys are model names and values are
    dictionaries containing evaluation metrics (mse, l2_loss, h1_loss).
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model in models.values():
        model.to(device)
        model.eval()

    results = {
        name: {
            "mse": 0.0,
            "l2_loss": 0.0,
            "h1_loss": 0.0,
        }
        for name in models.keys()
    }

    N = len(dataset)
    mse = torch.nn.MSELoss()
    l2 = LpLoss(d=2, p=2)
    h1 = H1Loss(d=2)
    
    with torch.no_grad():
        for i in tqdm(range(N), desc="Evaluating", unit="sample"):
            data = dataset[i]
            x = data["x"].unsqueeze(0).to(device)
            y_true = data["y"].unsqueeze(0).to(device)

            for name, model in models.items():
                y_pred = model(x)
                results[name]["mse"] += mse(y_pred, y_true).item()
                results[name]["l2_loss"] += l2(y_pred, y_true).item()
                results[name]["h1_loss"] += h1(y_pred, y_true).item()
        
        for name in results.keys():
            results[name]["mse"] /= N
            results[name]["l2_loss"] /= N
            results[name]["h1_loss"] /= N
    
    if print_results:
        for name, metrics in results.items():
            print("-" * 40)
            print(f"Average results for model '{name}':")
            print(f"  MSE:     {metrics['mse']:.6f}")
            print(f"  L2 Loss: {metrics['l2_loss']:.6f}")
            print(f"  H1 Loss: {metrics['h1_loss']:.6f}")

    return results

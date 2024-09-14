import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred):
    """
    Custom function to calculate accuracy, F1 score, AUC-ROC, precision, and recall.
    
    Args:
    - y_true (torch.Tensor): Ground truth labels (binary 0 or 1).
    - y_pred (torch.Tensor): Predicted logits or probabilities (before thresholding).
    
    Returns:
    - metrics_dict (dict): Dictionary containing calculated metrics.
    """
    
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    # print("original",
    #       y_pred,
    #       y_true)
    # print("np",
    #       y_pred_np,
    #       y_true_np)
    # Calculate metrics
    accuracy = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np)
    precision = precision_score(y_true_np, y_pred_np)
    recall = recall_score(y_true_np, y_pred_np)

    # AUC-ROC requires probabilities, not binary predictions
    if len(torch.unique(y_true)) > 1:  # AUC-ROC requires both classes to be present
        aucroc = roc_auc_score(y_true_np, y_pred.cpu().numpy())
    else:
        aucroc = float('nan')  # Return NaN if AUC can't be calculated

    # Create a dictionary of the metrics
    metrics_dict = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'aucroc': aucroc
    }

    return metrics_dict
from sklearn.model_selection import KFold
import torch
import pytorch_lightning as pl
import utils.visualization as vis
from data.transforms import transform
import data.dataset as ds 
import data.dataloader as dl 
from model.model import resnet34
from config import *

from sklearn.model_selection import KFold
import pytorch_lightning as pl
import torch

def train_kfolds(model, dataset, k_folds, trainer_config):
    """
    Perform k-fold cross-validation on the given model and dataset.
    
    Args:
        model: The model to train.
        dataset: The dataset used for training.
        k_folds: The number of folds for cross-validation.
        trainer_config: Configuration dictionary for the PyTorch Lightning Trainer.
    """
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{k_folds}")

        # Split dataset into train and validation sets
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(dataset, 
                                                    sampler=train_sampler, 
                                                    batch_size=BATCH_SIZE)
        val_loader = torch.utils.data.DataLoader(dataset, 
                                                  sampler=val_sampler, 
                                                  batch_size=BATCH_SIZE)

        # Initialize trainer for each fold
        trainer = pl.Trainer(**trainer_config)

        # Fit model
        trainer.fit(model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader)

        # Retrieve metrics from trainer's callbacks
        train_metrics = trainer.callback_metrics
        val_results = trainer.validate(model, val_loader, verbose=False)
        
        # Safeguard against missing metrics
        fold_train_loss = train_metrics.get('train_loss', 'N/A')
        fold_train_accuracy = train_metrics.get('train_accuracy', 'N/A')
        fold_val_loss = val_results[0].get('val_loss', 'N/A')
        fold_val_accuracy = val_results[0].get('val_accuracy', 'N/A')

        # Print results with proper handling of 'N/A' cases
        print(f"Fold {fold + 1} Train Loss: {fold_train_loss}, Train Accuracy: {fold_train_accuracy}")
        print(f"Fold {fold + 1} Val Loss: {fold_val_loss}, Val Accuracy: {fold_val_accuracy}")

        # Save fold results
        fold_results.append({
            'fold': fold + 1,
            'train_loss': fold_train_loss,
            'train_accuracy': fold_train_accuracy,
            'val_loss': fold_val_loss,
            'val_accuracy': fold_val_accuracy
        })

    return fold_results



if __name__ == '__main__':
    dataset = ds.CustomImageDataset(label_path, image_path, transform=transform)
    model = resnet34()
    fold_results = train_kfolds(model=model, 
                                dataset=dataset,
                                k_folds=k_folds,
                                trainer_config=trainer_config)
    vis.print_fold_results(fold_results)
    print('train.py')
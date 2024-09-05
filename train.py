from sklearn.model_selection import KFold
import torch
import pytorch_lightning as pl
import utils.visualization as vis
from data.transforms import transform
import data.dataset as ds 
import data.dataloader as dl 
from model.model import resnet34
from config import *

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

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{k_folds}")

        # Split dataset into train and validation sets
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
        val_loader = torch.utils.data.DataLoader(dataset, sampler=val_sampler, batch_size=BATCH_SIZE)

        # Initialize trainer for each fold
        trainer = pl.Trainer(**trainer_config)

        # Fit model
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Optionally, save the model for each fold
        trainer.save_checkpoint(f"saved/models/model_fold_{fold}.ckpt")




if __name__ == '__main__':
    dataset = ds.CustomImageDataset(label_path, image_path, transform=transform)
    model = resnet34()
    train_kfolds(model=model, 
                 dataset=dataset,
                 k_folds=k_folds,
                 trainer_config=trainer_config)
    print('train.py')
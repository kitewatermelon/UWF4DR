from sklearn.model_selection import KFold
import torch
import pytorch_lightning as pl
from sklearn.model_selection import KFold
import itertools
import time

import utils.visualization as vis
import data.transforms  as trs
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
        print("train_metrics:"+ str(train_metrics))
        val_results = trainer.validate(model, val_loader, verbose=False)
                
        # train metrics
        fold_train_loss = train_metrics.get('train_loss', 'N/A')
        fold_train_accuracy = train_metrics.get('train_accuracy', 'N/A')
        fold_train_f1_score = train_metrics.get('train_f1_score', 'N/A')
        fold_train_aucroc = train_metrics.get('train_aucroc', 'N/A')
        fold_train_precision = train_metrics.get('train_precision', 'N/A')
        fold_train_recall = train_metrics.get('train_recall', 'N/A')

        # validation metrics
        fold_val_loss = val_results[0].get('val_loss', 'N/A')
        fold_val_accuracy = val_results[0].get('val_accuracy', 'N/A')
        fold_val_f1_score = val_results[0].get('val_f1_score', 'N/A')
        fold_val_aucroc = val_results[0].get('val_aucroc', 'N/A')
        fold_val_precision = val_results[0].get('val_precision', 'N/A')
        fold_val_recall = val_results[0].get('val_recall', 'N/A')

        # Print results with proper handling of 'N/A' cases
        print(f"Fold {fold + 1} Train Loss: {fold_train_loss}, Train Accuracy: {fold_train_accuracy}, "
            f"Train F1 Score: {fold_train_f1_score}, Train AUC-ROC: {fold_train_aucroc}, "
            f"Train Precision: {fold_train_precision}, Train Recall: {fold_train_recall}")

        print(f"Fold {fold + 1} Val Loss: {fold_val_loss}, Val Accuracy: {fold_val_accuracy}, "
            f"Val F1 Score: {fold_val_f1_score}, Val AUC-ROC: {fold_val_aucroc}, "
            f"Val Precision: {fold_val_precision}, Val Recall: {fold_val_recall}")

        # Save fold results for all metrics
        fold_results.append({
            'fold': fold + 1,
            'train_loss'        : float(fold_train_loss),
            'train_accuracy'    : float(fold_train_accuracy),
            'train_f1_score'    : float(fold_train_f1_score),
            'train_aucroc'      : float(fold_train_aucroc),
            'train_precision'   : float(fold_train_precision),
            'train_recall'      : float(fold_train_recall),
            'val_loss'          : float(fold_val_loss),
            'val_accuracy'      : float(fold_val_accuracy),
            'val_f1_score'      : float(fold_val_f1_score),
            'val_aucroc'        : float(fold_val_aucroc),
            'val_precision'     : float(fold_val_precision),
            'val_recall'        : float(fold_val_recall)
        })

    return fold_results



if __name__ == '__main__':
    tile_sizes = [128, 64, 32, 16, 8, 4]  # 128부터 시작하여 2배씩 줄어듦
    use_center_crop_options = [True, False]    # CenterCrop 사용 여부
    for k in [5]:
        # 모든 transform 조합을 반복
        for use_center_crop, tile_size in itertools.product(use_center_crop_options, tile_sizes):
        
            if not use_center_crop:
                gaussian_sigma_options = [None, 0.3, 0.5, 0.7]  # GaussianMultiplyTransform의 sigma 설정
            else:
                gaussian_sigma_options = [None]  # CenterCrop을 사용하는 경우 GaussianMultiplyTransform을 사용하지 않음

            for gaussian_sigma in gaussian_sigma_options:
                # 각 조건에 맞는 transform 리스트 생성
                transform_list = trs.grid_search_transform(
                    image_size=IMAGE_SIZE, 
                    use_center_crop=use_center_crop, 
                    gaussian_sigma=gaussian_sigma,
                    tile_size=tile_size
                )
                
                for transform in transform_list:
                    info = f'K{k}T{tile_size},C{use_center_crop},G{gaussian_sigma}'
                    print(info)
                    # 데이터셋 생성
                    dataset = ds.CustomImageDataset(label_path, image_path, transform=transform)
                    
                    # 모델 생성
                    model = resnet34()
                    
                    # 교차검증으로 학습
                    fold_results = train_kfolds(
                        model=model, 
                        dataset=dataset,
                        k_folds=k,
                        trainer_config=trainer_config
                    )
                    
                    # 결과 저장 및 출력
                    vis.save_and_print_fold_results(info, fold_results)
                    vis.save_transform(info)

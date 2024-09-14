from sklearn.model_selection import KFold
import torch
import pytorch_lightning as pl
from sklearn.model_selection import KFold
import itertools
import time
from torch.utils.data import ConcatDataset

import utils.visualization as vis
import data.transforms  as trs
import data.dataset as ds 
import data.dataloader as dl 
from model.model import resnet34
from config import *


def train_kfolds(model, train_set, test_set, k_folds, trainer_config):
    """
    Perform k-fold cross-validation on the given model and dataset.
    
    Args:
        model: The model to train.
        train_set: The training dataset.
        test_set: The testing dataset used for final evaluation.
        k_folds: The number of folds for cross-validation.
        trainer_config: Configuration dictionary for the PyTorch Lightning Trainer.
    """
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_set)):
        print(f"Fold {fold + 1}/{k_folds}")



        # Split train_set into train and validation sets
        train_sampler   = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler     = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader    = torch.utils.data.DataLoader(train_set, 
                                                    sampler=train_sampler, 
                                                    batch_size=BATCH_SIZE,
                                                    )
        val_loader      = torch.utils.data.DataLoader(train_set, 
                                                sampler=val_sampler, 
                                                batch_size=BATCH_SIZE,
                                                )

        # No sampling for test set, use entire test set for final evaluation
        test_loader     = torch.utils.data.DataLoader(test_set, 
                                                batch_size=BATCH_SIZE, 
                                                shuffle=False,
                                                )

        # Initialize trainer for each fold
        trainer = pl.Trainer(**trainer_config)

        # Fit model
        trainer.fit(model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader)

        # Retrieve metrics from trainer's callbacks for training and validation
        train_metrics = trainer.callback_metrics
        # print("train_metrics:", str(train_metrics))
        val_results = trainer.validate(model, val_loader, verbose=False)

        # Evaluate on the test set after each fold training
        test_results = trainer.test(model, dataloaders=test_loader, verbose=False)

        # Extract train, val, and test metrics (assuming the metrics have these keys)
        fold_train_loss      = train_metrics.get('train_loss', 'N/A')
        fold_train_accuracy  = train_metrics.get('train_accuracy', 'N/A')
        fold_train_f1_score  = train_metrics.get('train_f1_score', 'N/A')
        fold_train_aucroc    = train_metrics.get('train_aucroc', 'N/A')
        fold_train_precision = train_metrics.get('train_precision', 'N/A')
        fold_train_recall    = train_metrics.get('train_recall', 'N/A')

        fold_val_loss        = val_results[0].get('val_loss', 'N/A')
        fold_val_accuracy    = val_results[0].get('val_accuracy', 'N/A')
        fold_val_f1_score    = val_results[0].get('val_f1_score', 'N/A')
        fold_val_aucroc      = val_results[0].get('val_aucroc', 'N/A')
        fold_val_precision   = val_results[0].get('val_precision', 'N/A')
        fold_val_recall      = val_results[0].get('val_recall', 'N/A')

        fold_test_loss       = test_results[0].get('test_loss', 'N/A')
        fold_test_accuracy   = test_results[0].get('test_accuracy', 'N/A')
        fold_test_f1_score   = test_results[0].get('test_f1_score', 'N/A')
        fold_test_aucroc     = test_results[0].get('test_aucroc', 'N/A')
        fold_test_precision  = test_results[0].get('test_precision', 'N/A')
        fold_test_recall     = test_results[0].get('test_recall', 'N/A')

        # Print fold results
        print(f"Fold {fold + 1} Train Loss: {fold_train_loss}, Train Accuracy: {fold_train_accuracy}")
        print(f"Fold {fold + 1} Val Loss: {fold_val_loss}, Val Accuracy: {fold_val_accuracy}")
        print(f"Fold {fold + 1} Test Loss: {fold_test_loss}, Test Accuracy: {fold_test_accuracy}")
        
        print(f""""
                fold_test_loss : {fold_test_loss}
                fold_test_accuracy : {fold_test_accuracy}
                fold_test_f1_score : {fold_test_f1_score}
                fold_test_aucroc : {fold_test_aucroc}
                fold_test_precision : {fold_test_precision}
                fold_test_recall : {fold_test_recall}
              """)

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
            'val_recall'        : float(fold_val_recall),
            'test_loss'         : float(fold_test_loss),
            'test_accuracy'     : float(fold_test_accuracy),
            'test_f1_score'     : float(fold_test_f1_score),
            'test_aucroc'       : float(fold_test_aucroc),
            'test_precision'    : float(fold_test_precision),
            'test_recall'       : float(fold_test_recall)
        })

    return fold_results



if __name__ == '__main__':
    tile_sizes = [32, 16, 8, 4]  # 128부터 시작하여 2배씩 줄어듦
    use_center_crop_options = ['center', 'circular', False]    # CenterCrop 사용 여부
    entire_time_start = time.time()
        # 모든 transform 조합을 반복
    for use_center_crop, tile_size in itertools.product(use_center_crop_options, tile_sizes):
        for t in [2]:    
            # if not use_center_crop:
            #     gaussian_sigma_options = [None, 50, 70]  # GaussianMultiplyTransform의 sigma 설정
            # else:
            gaussian_sigma_options = [None]  # CenterCrop을 사용하는 경우 GaussianMultiplyTransform을 사용하지 않음

            for gaussian_sigma in gaussian_sigma_options:
                for aug in [False, True]:
                    transform_list = trs.grid_search_transform(
                        image_size=IMAGE_SIZE, 
                        use_center_crop=use_center_crop, 
                        gaussian_sigma=gaussian_sigma,
                        tile_size=tile_size
                    )
                    
                    info = f'Fold{k_folds},Tile{tile_size},Crop{use_center_crop}task{t}aug{aug}'
                    print(info)
                    train_set   = ds.CustomImageDataset(label_path, 
                                                            image_path, 
                                                            transform_list[0], 
                                                            task=t)
                    if aug:
                        aug_train_set   = ds.CustomImageDataset(label_path, 
                                                                image_path, 
                                                                transform_list[0], 
                                                                task=t)
                        train_set = ConcatDataset([train_set, aug_train_set])

                    test_set        = ds.CustomImageDataset(test_label_path, 
                                                            test_image_path, 
                                                            transform_list[0], 
                                                            task=t)
                    print(train_set.__len__())
                    model = resnet34()
                    
                    fold_results = train_kfolds(
                        model=model,
                        train_set=train_set,
                        test_set=test_set,
                        k_folds=k_folds,
                        trainer_config=trainer_config,
                    )

                    # vis.visualize_RGB(dataset=dataset, rows=3)

                    # 결과 저장 및 출력
                    vis.save_and_print_fold_results(info, fold_results)
                    vis.save_transform(info)
    entire_time_end = time.time()
    print('entire time  :', entire_time_start - entire_time_end)
    print('time starts  :', entire_time_start)
    print('time ends    :', entire_time_end)
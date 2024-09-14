import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd 
import os
import shutil
import sys
sys.path.append('c:/Users/Administrator/dr박연수/')

from config import * 

CLA_label = {
    0: 'Normal',
    1: 'DR',
}

def visualize_RGB(dataset, rows):
    '''
        this function shows image devide RGB channel.
        dataset : a dataset contains images.
        rows : the number of images.

    '''
    # 이미지 그리드 설정
    figure = plt.figure(figsize=(12, 12))
    cols = 4
    for row in range(1, rows + 1):
        # 무작위로 이미지 선택
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        print(f"Original image shape: {img.shape}")
        
        # 이미지 numpy 배열로 변환
        img_np = img.numpy().transpose((1, 2, 0))

        # 이미지 RGB 채널 분리
        R = img_np[:, :, 0]
        G = img_np[:, :, 1]
        B = img_np[:, :, 2]

        # 각 채널 및 원본 이미지를 시각화
        figure.add_subplot(rows, cols, (row - 1) * cols + 1)
        plt.title(CLA_label[label])
        plt.axis("off")
        plt.imshow(np.clip(img_np, 0, 1))

        figure.add_subplot(rows, cols, (row - 1) * cols + 2)
        plt.title("Red Channel")
        plt.axis("off")
        plt.imshow(R, cmap="Reds")

        figure.add_subplot(rows, cols, (row - 1) * cols + 3)
        plt.title("Green Channel")
        plt.axis("off")
        plt.imshow(G, cmap="Greens")

        figure.add_subplot(rows, cols, (row - 1) * cols + 4)
        plt.title("Blue Channel")
        plt.axis("off")
        plt.imshow(B, cmap="Blues")
        
    plt.suptitle('Retinopathy Images and RGB Channels', y=0.92)
    plt.show()

def visualize_data(dataset, hist=False):
    '''
        This function shows random images from the dataset.
        dataset : a dataset contains images.
        hist : if True, display histogram alongside the image.
    '''

    sample_idx_list = []
    cols, rows = 4, 4
    for i in range(cols * rows ):
        sample_idx_list.append(torch.randint(len(dataset), size=(1,)).item())
    figure = plt.figure(figsize=(12, 12))
    for i, sample_idx in enumerate(sample_idx_list, 1):
        img, label = dataset[sample_idx]

        # Add subplot for the image
        figure.add_subplot(rows, cols, i)
        plt.title(CLA_label[label])
        plt.axis("off")
        img_np = img.numpy().transpose((1, 2, 0))
        img_valid_range = np.clip(img_np, 0, 1)  # Clip pixel values to [0, 1]
        plt.imshow(img_valid_range)
    
    plt.suptitle('Retinopathy Images', y=0.95)
    plt.show()

    # If hist is True, display histograms for pixel intensity distribution
    if hist:
        figure_hist = plt.figure(figsize=(12, 12))
        for i, sample_idx in enumerate(sample_idx_list, 1):
            img, label = dataset[sample_idx]
            img_np = img.numpy().transpose((1, 2, 0))
            img_valid_range = np.clip(img_np, 0, 1)

            # Add subplot for the histogram
            ax = figure_hist.add_subplot(rows, cols, i)
            ax.hist(img_valid_range.ravel(), bins=256, range=[0, 1])
            ax.set_title(CLA_label[label])
            ax.set_xlim([0, 1]) 
            ax.set_ylim([0, 300]) 

        plt.suptitle('Pixel Intensity Histograms', y=0.95)
        plt.show()
        
        
def visualize_data(dataset, hist=False):
    '''
        This function shows random images from the dataset.
        dataset : a dataset contains images.
        hist : if True, display histogram alongside the image.
    '''

    sample_idx_list = []
    cols, rows = 4, 4
    for i in range(cols * rows ):
        sample_idx_list.append(torch.randint(len(dataset), size=(1,)).item())
    figure = plt.figure(figsize=(12, 12))
    for i, sample_idx in enumerate(sample_idx_list, 1):
        img, label = dataset[sample_idx]

        # Add subplot for the image
        figure.add_subplot(rows, cols, i)
        plt.title(CLA_label[label])
        plt.axis("off")
        img_np = img.numpy().transpose((1, 2, 0))
        img_valid_range = np.clip(img_np, 0, 1)  # Clip pixel values to [0, 1]
        plt.imshow(img_valid_range)
    
    plt.suptitle('Retinopathy Images', y=0.95)
    plt.show()

    # If hist is True, display histograms for pixel intensity distribution
    if hist:
        figure_hist = plt.figure(figsize=(12, 12))
        for i, sample_idx in enumerate(sample_idx_list, 1):
            img, label = dataset[sample_idx]
            img_np = img.numpy().transpose((1, 2, 0))
            img_valid_range = np.clip(img_np, 0, 1)

            # Add subplot for the histogram
            ax = figure_hist.add_subplot(rows, cols, i)
            ax.hist(img_valid_range.ravel(), bins=256, range=[0, 1])
            ax.set_title(CLA_label[label])
            ax.set_xlim([0, 1]) 
            ax.set_ylim([0, 300]) 

        plt.suptitle('Pixel Intensity Histograms', y=0.95)
        plt.show()
        


def save_and_print_fold_results(info, fold_results):
    df = pd.DataFrame(fold_results)
    save_version = get_save_version(save_dir)

    expected_columns = ['Fold', 
                        'train_loss' ,    
                        'train_accuracy' ,
                        'train_f1_score' ,
                        'train_aucroc' ,  
                        'train_precision',
                        'train_recall' ,
                        'val_loss' ,    
                        'val_accuracy' ,
                        'val_f1_score' ,
                        'val_aucroc' ,  
                        'val_precision',
                        'val_recall' ,  
                        
                        'test_loss' ,    
                        'test_accuracy' ,
                        'test_f1_score' ,
                        'test_aucroc' ,  
                        'test_precision',
                        'test_recall' ,  
                        ]

    if len(df.columns) != len(expected_columns):
        raise ValueError(f"Length mismatch: Expected axis has {len(df.columns)} elements, new values have {len(expected_columns)} elements")
    
    df.columns = expected_columns
    
    print("\nFold Results:")
    print(df.to_string(index=False, formatters={
        'train_loss'        : '{:.4f}'.format,    
        'train_accuracy'    : '{:.4f}'.format,
        'train_f1_score'    : '{:.4f}'.format,
        'train_aucroc'      : '{:.4f}'.format,  
        'train_precision'   : '{:.4f}'.format,
        'train_recall'      : '{:.4f}'.format,

        'val_loss'      : '{:.4f}'.format,    
        'val_accuracy'  : '{:.4f}'.format,
        'val_f1_score'  : '{:.4f}'.format,
        'val_aucroc'    : '{:.4f}'.format,  
        'val_precision' : '{:.4f}'.format,
        'val_recall'    : '{:.4f}'.format,  

        'test_loss'      : '{:.4f}'.format,    
        'test_accuracy'  : '{:.4f}'.format,
        'test_f1_score'  : '{:.4f}'.format,
        'test_aucroc'    : '{:.4f}'.format,  
        'test_precision' : '{:.4f}'.format,
        'test_recall'    : '{:.4f}'.format,                                                 
        }
        ))
    
    mean_idx = len(df)
    std_idx = len(df)+1

    mean_values = df.iloc[:mean_idx, 1:].mean()  # Fold 값을 제외한 열의 평균 계산
    std_values = df.iloc[:mean_idx, 1:].std()    # Fold 값을 제외한 열의 표준편차 계산

    # 6번째 행을 'mean', 7번째 행을 'std'로 변경
    df.loc[mean_idx, 'Fold'] = 'mean'
    df.loc[std_idx, 'Fold'] = 'std'

    # 평균값과 표준편차 값을 데이터프레임에 반영
    df.iloc[mean_idx, 1:] = mean_values
    df.iloc[std_idx, 1:] = std_values

    version_dir = os.path.join(save_dir, f'version_{save_version}/fold_result')
    create_directory(version_dir)
    csv_file_path = os.path.join(version_dir, f'{info}.csv')
    df.to_csv(csv_file_path, index=False)

def save_transform(info, file_name="/transform_info.txt"):
    save_version = get_save_version(save_dir)

    version_dir = os.path.join(save_dir, f'version_{save_version}/fold_result')
    create_directory(version_dir)

    with open(version_dir+file_name, "a") as file:
        file.write(info + "\n")

def get_save_version(save_dir):
    version_dirs = [d for d in os.listdir(save_dir) if d.startswith('version_')]
    if version_dirs:
        version_numbers = [int(d.split('_')[1]) for d in version_dirs]
        save_version = max(version_numbers)
    else:
        save_version = 0

    return save_version

def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

if __name__=='__main__':
    save_transform()
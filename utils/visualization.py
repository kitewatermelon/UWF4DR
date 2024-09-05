import matplotlib.pyplot as plt
import numpy as np
import torch

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

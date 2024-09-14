import torch
from torch.utils.data import Dataset 
import os
from PIL import Image
import pandas as pd

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, 
                 transform=None, 
                 target_transform=None,
                 task=2):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir    = img_dir
        self.task       = task
        self.transform        = transform
        self.target_transform = target_transform
        
        # task에 따라 사용할 컬럼명 설정
        if self.task == 2:
            self.label_column = 'referable diabetic retinopathy'
        elif self.task == 3:
            self.label_column = 'diabetic macular edema'
        else:
            raise ValueError("Invalid task value. Task must be 2 or 3.")
        
        self.img_labels = self.img_labels.dropna(subset=[self.label_column])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx][self.label_column]
        if self.task == 2:
            label = float(label)
        
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
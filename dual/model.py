import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image
import numpy as np
import cv2
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

# IMAGE_SIZE 상수 정의
IMAGE_SIZE = 224

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    def __call__(self, img):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img)
        l = self.clahe.apply(l)
        img = cv2.merge((l, a, b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)

class GammaCorrection:
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        
    def __call__(self, img):
        img = np.array(img)
        img = img / 255.0
        img = np.power(img, self.gamma)
        img = np.uint8(img * 255)
        return Image.fromarray(img)



# DualResNet 클래스 정의
class DualResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(DualResNet, self).__init__()
        # 모델 정의
        weights = ResNet18_Weights.DEFAULT
        self.model1 = resnet18(weights=weights)
        self.model2 = resnet18(weights=weights)
        
        # 마지막 FC 레이어를 제거 (feature extraction만)
        self.model1 = nn.Sequential(*list(self.model1.children())[:-1])
        self.model2 = nn.Sequential(*list(self.model2.children())[:-1])
        
        # 결합된 특징으로부터 최종 예측을 위한 FC 레이어
        self.fc = nn.Linear(512 * 2, num_classes)
    
    def forward(self, img1, img2):
        # 각각의 ResNet 모델을 통해 특징 추출
        feat1 = self.model1(img1)
        feat2 = self.model2(img2)
        
        # (batch_size, 512, 1, 1) 형태의 출력을 (batch_size, 512)로 변환
        feat1 = feat1.view(feat1.size(0), -1)
        feat2 = feat2.view(feat2.size(0), -1)
        
        # 두 특징을 결합
        combined_features = torch.cat((feat1, feat2), dim=1)
        
        # 결합된 특징을 통해 최종 예측
        out = self.fc(combined_features)
        
        return out
# 모델 클래스를 정의합니다.
class model:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # 첫 번째 트랜스폼: 원본 이미지에 대한 트랜스폼
        self.transform1 = transforms.Compose([
            transforms.Resize((IMAGE_SIZE//2, IMAGE_SIZE//2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            CLAHETransform(clip_limit=5.0, tile_grid_size=(8, 8)),
            GammaCorrection(gamma=0.45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform2 = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.CenterCrop((IMAGE_SIZE // 2, IMAGE_SIZE // 2)),  
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            CLAHETransform(clip_limit=5.0, tile_grid_size=(8, 8)),
            GammaCorrection(gamma=0.45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def init(self):
        # 모델 초기화
        self.model = DualResNet(num_classes=1)
        self.model = self.model.to(self.device)
    
    def load(self):
        model_path = os.path.join(os.path.dirname(__file__), 'final_model.pth')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    
    def predict(self, image_path):
        # 이미지를 받아서 예측을 수행
        img = Image.open(image_path).convert('RGB')
        
        img1 = self.transform1(img).unsqueeze(0).to(self.device)
        img2 = self.transform2(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img1, img2).squeeze(1)  # 두 이미지에 대한 예측
            predicted = torch.sigmoid(output).item()
            return 1 if predicted > 0.5 else 0  # 이진 분류 결과 반환
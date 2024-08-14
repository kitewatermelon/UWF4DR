import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import cv2

# IMAGE_SIZE 상수 정의
IMAGE_SIZE = 256

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
    
# 가우시안 분포 이미지를 생성하는 함수
def create_gaussian_mask(size, sigma=0.5):
    """ size: (H, W) 형태의 튜플로 이미지 크기 설정
        sigma: 가우시안 분포의 표준편차
    """
    H, W = size
    # 가우시안 분포를 만드는 x, y 좌표계 생성
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    x, y = np.meshgrid(x, y)
    
    # 2차원 가우시안 분포 계산
    d = np.sqrt(x*x + y*y)
    gaussian = np.exp(-d**2 / (2.0 * sigma**2))
    
    return gaussian

# MultiplyTransform 클래스 수정: 가우시안 분포 이미지 곱하기
class GaussianMultiplyTransform:
    def __init__(self, size, sigma=0.5):
        self.gaussian_mask = create_gaussian_mask(size, sigma)
        
    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)
        
        # 가우시안 마스크를 채널에 맞게 확장
        gaussian_mask = np.expand_dims(self.gaussian_mask, axis=-1)
        gaussian_mask = np.repeat(gaussian_mask, img_np.shape[-1], axis=-1)
        
        # 이미지에 가우시안 마스크 곱하기
        img_np *= gaussian_mask
        
        # 이미지를 다시 uint8로 변환
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

class model:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.CenterCrop((IMAGE_SIZE - 20 ,IMAGE_SIZE -20 )),  # // 사용으로 정수 나누기
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                        
                CLAHETransform(clip_limit=5.0, tile_grid_size=(8, 8)),
                GammaCorrection(gamma=0.45),
                GaussianMultiplyTransform(size=(IMAGE_SIZE, IMAGE_SIZE), sigma=0.8),  # 가우시안 분포 곱하기
                
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    def init(self):
        self.model = resnet18(pretrained=True)  # 최신 torchvision 사용 시
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  # Assuming binary classification
        self.model = self.model.to(self.device)
        self.model.eval()

    def load(self, weights_path):
        if self.model is None:
            self.init()
        model_weights_path = os.path.join(weights_path, 'model_weights.pth')  # Assuming the weights file is named 'model_weights.pth'
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))

    def predict(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img)
            probs = torch.softmax(output, dim=1)
            pred = probs.argmax(dim=1).item()
            pred = 1 - pred  # Reverse the prediction
        return pred

# Ensure to save the model weights file (model_weights.pth) in the same directory as this model.py file.

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet34
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



def create_doughnut_gaussian_mask(size, sigma_outer, sigma_inner):
    center = size[0] // 2, size[1] // 2
    x = np.linspace(0, size[0] - 1, size[0])
    y = np.linspace(0, size[1] - 1, size[1])
    X, Y = np.meshgrid(x, y)
    
    # Calculate distances from center
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # Gaussian masks
    gaussian_outer = np.exp(-dist_from_center**2 / (2 * sigma_outer**2))
    gaussian_inner = np.exp(-dist_from_center**2 / (2 * sigma_inner**2))
    
    # Doughnut mask: subtract inner Gaussian from outer Gaussian
    doughnut_mask = gaussian_outer - gaussian_inner
    doughnut_mask = np.clip(doughnut_mask, 0, 1)
    
    return doughnut_mask


# MultiplyTransform 클래스 수정: 가우시안 분포 이미지 곱하기
class GaussianMultiplyTransform:
    def __init__(self, size, sigma=0.5):
        self.gaussian_mask = create_doughnut_gaussian_mask(size, sigma_inner=0, sigma_outer=50)
        
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
                
        GaussianMultiplyTransform(size=(IMAGE_SIZE, IMAGE_SIZE)),  # 가우시안 분포 곱하기
        CLAHETransform(clip_limit=5.0, tile_grid_size=(8, 8)),
        GammaCorrection(gamma=0.45),
        transforms.CenterCrop(IMAGE_SIZE - 50),  # // 사용으로 정수 나누기
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

    def init(self):
        self.model = resnet34(pretrained=True)  # 최신 torchvision 사용 시
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

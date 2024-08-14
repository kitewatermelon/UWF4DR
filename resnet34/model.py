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

class ErodeTransform:
    def __init__(self, kernel_size=(3, 3), iterations=1):
        self.kernel = np.ones(kernel_size, np.uint8)
        self.iterations = iterations

    def __call__(self, img):
        img = np.array(img)
        img = cv2.erode(img, self.kernel, iterations=self.iterations)
        return Image.fromarray(img)

class DilateTransform:
    def __init__(self, kernel_size=(3, 3), iterations=1):
        self.kernel = np.ones(kernel_size, np.uint8)
        self.iterations = iterations

    def __call__(self, img):
        img = np.array(img)
        img = cv2.diate(img, self.kernel, iterations=self.iterations)
        return Image.fromarray(img)
        
class CropLargestSquareTransform:
    def __call__(self, img):
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 127, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return Image.fromarray(img)
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        side_length = max(w, h)
        x_center = x + w // 2
        y_center = y + h // 2
        x1 = max(x_center - side_length // 2, 0)
        y1 = max(y_center - side_length // 2, 0)
        x2 = min(x_center + side_length // 2, img.shape[1])
        y2 = min(y_center + side_length // 2, img.shape[0])
        
        img_cropped = img[y1:y2, x1:x2]
        return Image.fromarray(img_cropped)

class LaplacianTransform:
    def __init__(self):
        # 라플라시안 마스크를 정의합니다.
        self.mask = np.array([[1, 1, 1], 
                              [1, -8, 1], 
                              [1, 1, 1]])

    def __call__(self, original):
        # PIL 이미지를 numpy 배열로 변환
        img = np.array(original)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 라플라시안 필터 적용
        laplacian = cv2.filter2D(img, -1, self.mask)
        img = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        # img = 1 - img
        img = original-img
        # numpy 배열을 다시 PIL 이미지로 변환
        return Image.fromarray(img)
    
class OtsuThresholdTransform:
    def __call__(self, original):
        # PIL 이미지를 numpy 배열로 변환
        img = np.array(original)

        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 오츠 이진화 적용
        _, otsu_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 이진화된 이미지를 BGR로 변환 (필요한 경우)
        otsu_img_bgr = cv2.cvtColor(otsu_img, cv2.COLOR_GRAY2BGR)
        img = img - otsu_img_bgr
        # numpy 배열을 다시 PIL 이미지로 변환
        return Image.fromarray(img)


class model:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop((IMAGE_SIZE // 2,IMAGE_SIZE// 2 )),  
        
        LaplacianTransform(),  # 가우시안 블러 추가
        
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        
        CLAHETransform(clip_limit=5.0, tile_grid_size=(8, 8)),
        GammaCorrection(gamma=0.45),
        
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

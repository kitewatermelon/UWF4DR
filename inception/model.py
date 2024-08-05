import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import cv2

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

class model:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        CLAHETransform(clip_limit=5.0, tile_grid_size=(8, 8)),
        GammaCorrection(gamma=0.45),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
       ])

    def init(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

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

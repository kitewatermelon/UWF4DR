import cv2
import numpy as np
from PIL import Image 
import torchvision.transforms as transforms 
from config import * 

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
    
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    gaussian_outer = np.exp(-dist_from_center**2 / (2 * sigma_outer**2))
    gaussian_inner = np.exp(-dist_from_center**2 / (2 * sigma_inner**2))
    
    doughnut_mask = gaussian_outer - gaussian_inner
    doughnut_mask = np.clip(doughnut_mask, 0, 1)
    
    return doughnut_mask


class GaussianMultiplyTransform:
    def __init__(self, size, sigma_inner=30, sigma_outer=100):
        self.gaussian_mask = create_doughnut_gaussian_mask(size, sigma_inner, sigma_outer)
        
    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)
        
        gaussian_mask = np.expand_dims(self.gaussian_mask, axis=-1)
        gaussian_mask = np.repeat(gaussian_mask, img_np.shape[-1], axis=-1)
        
        img_np *= gaussian_mask
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)
    

def grid_search_transform(image_size, 
                          use_center_crop=False, 
                          gaussian_sigma=None, 
                          tile_size=4):
    transform_list = []
    
    # 기본적인 transforms
    current_transform = [
        transforms.Resize((image_size, image_size)),
        CLAHETransform(clip_limit=5.0, tile_grid_size=(tile_size, tile_size)),
        GammaCorrection(gamma=0.35)
    ]
    
    # 옵션 1: CenterCrop 적용
    if use_center_crop:
        current_transform.append(transforms.CenterCrop(image_size // 2))
    
    # 옵션 2 & 3: GaussianMultiplyTransform 적용
    if gaussian_sigma:
        current_transform.append(GaussianMultiplyTransform(size=(image_size, image_size), sigma_inner=gaussian_sigma, sigma_outer=0))
    
    # 마지막 기본 변환들 추가
    current_transform.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_list.append(transforms.Compose(current_transform))
    
    return transform_list
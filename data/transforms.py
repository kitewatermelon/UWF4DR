import cv2
import numpy as np
from PIL import Image 
import torchvision.transforms as transforms 
from config import * 
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

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
    
# class CLAHETransform:
#     def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
#         self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
#     def __call__(self, img):
#         img = np.array(img)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         b, g, r = cv2.split(img)
        
#         # 녹색 채널에만 CLAHE 적용
#         g = self.clahe.apply(g)
        
#         img = cv2.merge((b, g, r))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return Image.fromarray(img)

class GammaCorrection:
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        
    def __call__(self, img):
        img = np.array(img)
        img = img / 255.0
        img = np.power(img, self.gamma)
        img = np.uint8(img * 255)
        return Image.fromarray(img)
    
# class GammaCorrection:
#     def __init__(self, gamma=1.0):
#         self.gamma = gamma
        
#     def __call__(self, img):
#         img = np.array(img)
#         img = img / 255.0
#         img = np.power(img, self.gamma)
#         img = np.uint8(img * 255)
        
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         b, g, r = cv2.split(img)
        
#         # 녹색 채널에만 Gamma Correction 적용
#         g = np.power(g / 255.0, self.gamma) * 255.0
#         g = np.uint8(g)
        
#         img = cv2.merge((b, g, r))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return Image.fromarray(img)

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
    
from PIL import Image, ImageDraw

class CircularCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # 이미지 크기 및 중심 계산
        w, h = img.size
        cx, cy = w // 2, h // 2

        # 원형 마스크 생성
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((cx - self.size // 2, cy - self.size // 2, cx + self.size // 2, cy + self.size // 2), fill=255)

        # 마스크 적용
        img = np.array(img)
        mask = np.array(mask)
        img = np.where(mask[:, :, None] == 255, img, 0)
        
        # 크롭된 이미지로 변환
        img = Image.fromarray(img)
        img = img.crop((cx - self.size // 2, cy - self.size // 2, cx + self.size // 2, cy + self.size // 2))
        return img










def grid_search_transform(image_size, 
                          use_center_crop=False, 
                          gaussian_sigma=None, 
                          aug=True,
                          tile_size=4):
    
    # 기본적인 transforms
    aug_transform = [
        transforms.Resize((image_size, image_size)),
        CLAHETransform(clip_limit=5.0, tile_grid_size=(tile_size, tile_size)),
        GammaCorrection(gamma=0.35)
    ]
    original_transform = [
        transforms.Resize((image_size, image_size)),
        CLAHETransform(clip_limit=5.0, tile_grid_size=(tile_size, tile_size)),
        GammaCorrection(gamma=0.35)
    ]
    
    if aug:
        aug_transform.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
        ])

    # 옵션 1: CenterCrop 적용
    if use_center_crop == 'center':
        original_transform.append(transforms.CenterCrop(image_size // 2))
        aug_transform.append(transforms.CenterCrop(image_size // 2))
    if use_center_crop == 'circular':
        original_transform.append(CircularCrop(IMAGE_SIZE-10))
        aug_transform.append(CircularCrop(IMAGE_SIZE-10))


    # 옵션 2 & 3: GaussianMultiplyTransform 적용
    if gaussian_sigma:
        original_transform.append(GaussianMultiplyTransform(size=(image_size, image_size), sigma_inner=gaussian_sigma, sigma_outer=0))
        aug_transform.append(GaussianMultiplyTransform(size=(image_size, image_size), sigma_inner=gaussian_sigma, sigma_outer=0))
        
    # 마지막 기본 변환들 추가
    original_transform.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    aug_transform.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_list = [
        transforms.Compose(original_transform), # for orignal
        transforms.Compose(aug_transform),      # for augmentation
        ]
    print(transform_list[0],
          transform_list[1],
          )
    return transform_list


transform = grid_search_transform(IMAGE_SIZE, 
                          use_center_crop=False, 
                          gaussian_sigma=None, 
                          aug=True,
                          tile_size=4)

experimental = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])
    
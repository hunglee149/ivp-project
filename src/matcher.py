import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import numpy as np
import cv2
from typing import Tuple, Dict

class FeatureMatcher:
    def __init__(
        self, 
        max_num_keypoints: int = 2048, 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_image_size: int = 1024
    ):
        self.device = device
        self.max_image_size = max_image_size
        
        print(f"Khoi tao SuperPoint & LightGlue tren {device}...")
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
        self.matcher = LightGlue(features='superpoint').eval().to(device)

    def load_image(self, image_path: str) -> Tuple[torch.Tensor, float]:
        img_tensor = load_image(image_path)
        
        _, h, w = img_tensor.shape
        max_dim = max(h, w)
        scale = 1.0
        
        if max_dim > self.max_image_size:
            scale = self.max_image_size / max_dim
            new_h, new_w = int(h * scale), int(w * scale)
            
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_resized = cv2.resize(img_np, (new_w, new_h))
            
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            
        return img_tensor.to(self.device), scale

    def match_pair(self, image0: torch.Tensor, image1: torch.Tensor):
        with torch.no_grad():
            feats0 = self.extractor.extract(image0)
            feats1 = self.extractor.extract(image1)
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
            
        return feats0, feats1, matches01

    def match_images(self, image_path0: str, image_path1: str):
        image0, scale0 = self.load_image(image_path0)
        image1, scale1 = self.load_image(image_path1)
        
        feats0, feats1, matches01 = self.match_pair(image0, image1)
        
        matches = matches01['matches']
        
        points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
        points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()
        
        points0 = points0 / scale0
        points1 = points1 / scale1
        
        return points0, points1
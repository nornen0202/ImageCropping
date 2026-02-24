import os
import io
import torch
import numpy as np
from PIL import Image
import open_clip

class ClipFeatureExtractor:
    """
    OpenCLIP(ViT-L/14) 기반 이미지 & 텍스트 특징 추출기
    - 주어진 PIL Image 또는 텍스트 리스트에 대해 임베딩을 반환합니다.
    """
    def __init__(self, model_name=None, pretrained=None, device=None, priority="high_efficiency"):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.priority = priority
        
        if model_name is None:
            model_name = "ViT-H-14" if priority == "quality_first" else "ViT-L-14"
        if pretrained is None:
            pretrained = "laion2b_s32b_b79k" if priority == "quality_first" else "openai"
            
        print(f"[C1 CLIP] Loading OpenCLIP {model_name} ({pretrained}) on {self.device} (Mode: {self.priority})...")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # 모델을 평가 모드로 전환하고, 메모리 이점을 위해 FP16 변환 시도
        self.model.eval()
        if self.device != "cpu":
            self.model = self.model.half()
            
    @torch.no_grad()
    def encode_images(self, images: list):
        """
        images: list of PIL.Image
        반환: numpy array (N, embed_dim) FP16
        """
        if not images:
            return np.array([])
            
        tensors = []
        for img in images:
            # ensure RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # OpenCLIP Transform
            t = self.preprocess(img).unsqueeze(0)
            tensors.append(t)
            
        batch = torch.cat(tensors).to(self.device)
        
        # half precision if model is half
        if self.device != "cpu":
            batch = batch.half()
            
        image_features = self.model.encode_image(batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().astype(np.float16)

    @torch.no_grad()
    def encode_texts(self, texts: list):
        """
        texts: list of strings (태그나 캡션)
        반환: numpy array (N, embed_dim) FP16
        """
        if not texts:
            return np.array([])
            
        text_tokens = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().astype(np.float16)

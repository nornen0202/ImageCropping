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

        self.model, self.preprocess, self.tokenizer = self._load_with_fallback(
            model_name=model_name,
            pretrained=pretrained,
            device=self.device,
        )
        
        # 모델을 평가 모드로 전환하고, 메모리 이점을 위해 FP16 변환 시도
        self.model.eval()
        if self.device != "cpu":
            self.model = self.model.half()

    def _load_once(self, model_name: str, pretrained: str, device: str):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
        return model, preprocess, tokenizer

    def _load_with_fallback(self, model_name: str, pretrained: str, device: str):
        attempts = [(model_name, pretrained, device)]
        if str(device).startswith("cuda"):
            if model_name != "ViT-L-14":
                attempts.append(("ViT-L-14", "openai", device))
            attempts.append(("ViT-L-14", "openai", "cpu"))

        last_error = None
        for cand_model, cand_pretrained, cand_device in attempts:
            try:
                if last_error is not None:
                    print(
                        f"[C1 CLIP][retry] trying {cand_model} ({cand_pretrained}) on {cand_device} "
                        f"after: {type(last_error).__name__}"
                    )
                return self._load_once(cand_model, cand_pretrained, cand_device)
            except RuntimeError as exc:
                last_error = exc
                if "out of memory" not in str(exc).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("failed to initialize C1 CLIP extractor")
            
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

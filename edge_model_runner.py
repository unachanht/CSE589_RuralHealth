# edge_model_runner.py
import torch
import numpy as np
import cv2
import torch.nn as nn

class EdgeModelRunner:
    """
    Runs the quantized edge model:
    - segmentation head: mask logits (Bx1xHxW)
    - classification head: malignancy score (Bx1)
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    @torch.inference_mode()
    def infer(self, img_bgr: np.ndarray, mc_samples: int = 1):
        # Preprocess
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ten = torch.from_numpy(rgb).permute(2,0,1).float()[None] / 255.0
        ten = ten.to(self.device)

        seg_logits_list = []
        cls_logits_list = []

        # Optional MC dropout for segmentation uncertainty
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        for _ in range(mc_samples):
            seg_logits, cls_logits = self.model(ten)   # *** USES MODEL HERE ***
            seg_logits_list.append(seg_logits)
            cls_logits_list.append(cls_logits)

        seg_mean = torch.mean(torch.stack(seg_logits_list), dim=0)
        cls_mean = torch.mean(torch.stack(cls_logits_list), dim=0)

        # ROI mask: sigmoid > 0.5
        seg_mask = (torch.sigmoid(seg_mean)[0,0] > 0.5).cpu().numpy().astype(np.uint8)

        # Malignancy score
        mal_score = float(torch.sigmoid(cls_mean).item())

        # Priority scoring formula (your design)
        quality = cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        priority = mal_score * 0.6 + min(1.0, quality / 500.0) * 0.4

        return {
            "malignancy": mal_score,
            "mask": seg_mask,
            "priority": float(priority),
            "quality": float(min(1.0, quality / 500.0))
        }

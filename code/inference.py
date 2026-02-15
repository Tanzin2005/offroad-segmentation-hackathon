import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from PIL import Image
from torch import nn

# =============================
# Configuration
# =============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "segmentation_head.pth"
BACKBONE_SIZE = "small"

# Same mapping as training
value_map = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

n_classes = len(value_map)

# =============================
# Model Definition
# =============================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

# =============================
# Load Backbone
# =============================

def load_backbone():
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }

    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
    backbone.eval()
    backbone.to(DEVICE)

    return backbone

# =============================
# Preprocessing
# =============================

def get_transform(h, w):
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# =============================
# Inference Function
# =============================

def predict(image_path, output_path="prediction.png"):

    # Image size logic (same as training)
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    transform = get_transform(h, w)

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    backbone = load_backbone()

    with torch.no_grad():
        features = backbone.forward_features(input_tensor)["x_norm_patchtokens"]

    n_embedding = features.shape[2]

    model = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        logits = model(features)
        output = F.interpolate(
            logits,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )

    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Save colored mask
    colored_mask = (pred * 25).astype(np.uint8)
    cv2.imwrite(output_path, colored_mask)

    print(f"Prediction saved to {output_path}")


# =============================
# Run
# =============================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py path_to_image")
    else:
        predict(sys.argv[1])

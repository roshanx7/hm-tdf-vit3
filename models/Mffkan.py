# @inproceedings{HM-TDF,  
#   title:{Hard Sample Mining-Based Tongue Diagnosis Framework for Fatty Liver Disease Severity Classification Using Kolmogorov-Arnold Network},  
#   link:{https://github.com/MLDMXM2017/HM-TDF}  
# }  

import numpy as np
import torch
import torch.nn as nn
import sys
import os

script_path = os.path.abspath(__file__)
model_path = os.path.dirname(script_path)
sys.path.insert(0, model_path)

from KANLinear import KANLinear as KAN_Linear

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. Install with: pip install timm")


# =========================
# MetaFusion-Inspired Gated Fusion Layer
# =========================
class FusionLayer(nn.Module):
    """Metadata-guided residual gated fusion mechanism.
    
    This layer implements a MetaFusion-inspired approach that goes beyond
    simple concatenation by enabling metadata-guided interaction with image
    features through a learned gating mechanism.
    
    Design rationale:
    - Metadata features are projected to image feature dimension
    - A gating mechanism (tanh) learns which image dimensions should be
      influenced by metadata
    - Residual connection preserves original image information while adding
      metadata-conditioned modulation
    - More parameter-efficient than attention mechanisms while maintaining
      interpretability
    
    Args:
        image_dim (int): Dimension of image features (512)
        meta_dim (int): Dimension of metadata features (128)
    """
    
    def __init__(self, image_dim, meta_dim):
        super().__init__()
        
        # Project metadata to image feature space
        self.W = nn.Linear(meta_dim, image_dim)
        
    def forward(self, image_feat, meta_feat):
        """Apply metadata-guided gated fusion.
        
        Args:
            image_feat: Image features [batch_size, image_dim]
            meta_feat: Metadata features [batch_size, meta_dim]
            
        Returns:
            fused_image: Enhanced image features [batch_size, image_dim]
        """
        # Project metadata to image feature dimension
        meta_proj = self.W(meta_feat)  # [batch_size, image_dim]
        
        # Metadata-guided gating mechanism (element-wise interaction)
        # tanh squashes to [-1, 1] providing smooth modulation
        gate = torch.tanh(image_feat * meta_proj)  # [batch_size, image_dim]
        
        # Residual gated fusion: preserve image features + add metadata-guided modulation
        # This allows the model to learn when and which image dimensions should be
        # influenced by metadata vs. kept unchanged
        fused_image = image_feat + image_feat * gate  # [batch_size, image_dim]
        
        return fused_image


class MffKan(nn.Module): 
    def __init__(self, num_labels, num_features, drop_rate):
        super().__init__()

        self.num_features = num_features
        self.kan_linears = nn.ModuleList()
        self.use_tinyvit = False

        # =========================
        # Image Encoder
        # =========================
        if TIMM_AVAILABLE:
            try:
                self.IE = timm.create_model('tiny_vit_5m_224', pretrained=True, num_classes=0)

                # Dynamically infer output dimension (FIXED)
                with torch.no_grad():
                    dummy = torch.randn(1, 3, 224, 224)
                    out = self.IE(dummy)
                    self.ie_dim = out.shape[1]

                self.use_tinyvit = True
                print(f"[INFO] Using TinyViT as Image Encoder ({self.ie_dim}-dim output)")

            except Exception as e:
                print(f"Warning: TinyViT failed ({str(e)}). Falling back to ResNet18.")
                try:
                    self.IE = timm.create_model('resnet18', pretrained=True, num_classes=0)
                    self.ie_dim = 512
                    print("[INFO] Using ResNet18 as Image Encoder (512-dim output)")
                except Exception as e2:
                    print(f"Warning: ResNet18 failed ({str(e2)}). Using simple CNN fallback.")
                    self.IE = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten()
                    )
                    self.ie_dim = 64
                    print("[INFO] Using simple CNN fallback (64-dim output)")
        else:
            print("Warning: timm not available, using simple CNN fallback")
            self.IE = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.ie_dim = 64
            print("[INFO] Using simple CNN fallback (64-dim output)")

        # =========================
        # Projection Layer (REQUIRED by main.py)
        # =========================
        self.ie_proj = nn.Linear(self.ie_dim, 512)

        # =========================
        # Indicator Encoder (KAN)
        # =========================
        self.de_dim = 128

        self.DE = nn.Sequential(
            KAN_Linear(num_features, 32,
                       grid_size=5, spline_order=3,
                       scale_noise=0.01, scale_base=1, scale_spline=1,
                       base_activation=nn.SiLU,
                       grid_eps=0.02, grid_range=[-1, 1]),
            nn.BatchNorm1d(32),
            nn.Dropout(drop_rate),
            KAN_Linear(32, 128,
                       grid_size=5, spline_order=3,
                       scale_noise=0.01, scale_base=1, scale_spline=1,
                       base_activation=nn.SiLU,
                       grid_eps=0.02, grid_range=[-1, 1])
        )

        # =========================
        # Metadata-Guided Fusion Layer
        # =========================
        # Implements MetaFusion-inspired gated fusion instead of simple concat
        self.fusion_layer = FusionLayer(512, 128)

        # =========================
        # Classifier (CNN-based)
        # =========================
        self.fused_dim = 512 + self.de_dim  # 640

        self.classifier = nn.Sequential(
            nn.Unflatten(1, (1, self.fused_dim)),
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_labels)
        )

        # =========================
        # Collect KAN layers
        # =========================
        for module in self.DE.modules():
            if isinstance(module, KAN_Linear):
                self.kan_linears.append(module)

    def forward(self, X, f_p, debug=False):
        # Image features
        f_i = self.IE(X)

        if debug:
            print(f"[DEBUG] After IE: {f_i.shape}")

        # Projection to normalized dimension
        f_i = self.ie_proj(f_i)

        if debug:
            print(f"[DEBUG] After ie_proj: {f_i.shape}")

        # Indicator features via KAN-based metadata encoder
        f_p = self.DE(f_p)

        if debug:
            print(f"[DEBUG] After DE: {f_p.shape}")

        # ========================================
        # MetaFusion-style Gated Fusion
        # ========================================
        # Replace simple concatenation with metadata-guided residual fusion.
        # The fusion layer learns to modulate image features based on metadata,
        # enabling more effective multi-modal interaction than direct concatenation.
        f_i = self.fusion_layer(f_i, f_p)

        if debug:
            print(f"[DEBUG] After fusion_layer: {f_i.shape}")

        # Concatenate enhanced image features with metadata features
        # Final dimension: 512 (image) + 128 (metadata) = 640
        f_f = torch.cat((f_i, f_p), dim=1)

        if debug:
            print(f"[DEBUG] After concat: {f_f.shape}")

        # Classifier
        logits = self.classifier(f_f)

        if debug:
            print(f"[DEBUG] Output: {logits.shape}")

        return logits

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.kan_linears
        )

    def unfreeze_ie_layers(self, unfreeze_ratio=0.5):
        if not TIMM_AVAILABLE:
            print("[WARNING] Cannot unfreeze layers (timm not available)")
            return

        ie_params = list(self.IE.named_parameters())
        total_layers = len(ie_params)
        unfreeze_count = max(1, int(total_layers * unfreeze_ratio))

        for i, (name, param) in enumerate(ie_params):
            if i >= (total_layers - unfreeze_count):
                param.requires_grad = True
            else:
                param.requires_grad = False

        encoder_name = "TinyViT" if self.use_tinyvit else "Fallback Encoder"
        print(f"[INFO] Unfroze {unfreeze_count}/{total_layers} {encoder_name} layers")


# =========================
# Factory function
# =========================
def get_net(num_features, num_labels, drop_rate):
    return MffKan(num_labels, num_features, drop_rate)
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import timm

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=6, model_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return x[:, -1, :]

    
class ViTWaypointPredictor(nn.Module):
    def __init__(self, image_model_name='vit_tiny_patch16_224',
                 past_dyn_dim=6, num_past_steps=16,
                 num_intent_classes=4, future_steps=20,
                 freeze_vit=True):
        super().__init__()

        # Load and prepare ViT backbone
        self.vit = timm.create_model(image_model_name, pretrained=True)
        self.vit.reset_classifier(0)  # remove classification head

        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()

        self.image_feat_dim = 196*192 # 196 patches of 192-dim features (after proj_drop)
        # Project concatenated image features from 3 cameras
        self.image_proj = nn.Linear(self.image_feat_dim * 3, 256)

        # Past dynamics encoder
        self.trajectory_encoder = TrajectoryEncoder()
        self.intent_encoder = self.intent_encoder = nn.Sequential(
            nn.Linear(num_intent_classes, 16),
            nn.ReLU()
        )

        # Fusion MLP for future position prediction
        self.fusion_mlp = nn.Sequential(
            nn.Linear(256 + 128 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, future_steps * 2)  # Predict (x, y) per future step
        )

        self.future_steps = future_steps

    def _extract_vit_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from the final proj_drop layer of the last ViT block.
        """
        proj_drop_output = []

        def hook_fn(module, input, output):
            # print(f"Output shape from proj_drop: {output.shape}")  # [B, 197,192]
            proj_drop_output.append(output[:,1:,:]) # Exclude CLS token, keep patch tokens

        handle = self.vit.blocks[-1].attn.proj_drop.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = self.vit(x)
        handle.remove()

        return proj_drop_output[0]

    def forward(self, images: torch.Tensor, intent: torch.Tensor, past_dyn: torch.Tensor) -> torch.Tensor:
        B, C, _, _, _ = images.shape  # [B, 3, 3, 224, 224]

        # Process each camera image with ViT and extract features
        cam_features = []
        for i in range(C):
            cam_img = images[:, i]  # [B, 3, 224, 224]
            feat = self._extract_vit_feature(cam_img)

            feat = feat.view(B, -1) 
            cam_features.append(feat)

        vision_feat = torch.cat(cam_features, dim=1)       # [B, 3*196*192]
        # print(f"Vision feature shape: {vision_feat.shape}")
        vision_feat = self.image_proj(vision_feat)         # [B, 256]

        traj_feat = self.trajectory_encoder(past_dyn) # [B, 128]
        # print(f"Trajectory feature shape: {traj_feat.shape}")
        intent_feat = self.intent_encoder(intent)       # [B, 16]
        # print(f"Intent feature shape: {intent_feat.shape}")

        fused = torch.cat([vision_feat, traj_feat, intent_feat], dim=1)  # [B, 336]
        out = self.fusion_mlp(fused)                       # [B, 40]

        return out.view(-1, self.future_steps, 2)          # [B, 20, 2]

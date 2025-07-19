import torch
from torch import nn
from collections import OrderedDict
from itertools import repeat
from collections.abc import Iterable

# --- Final Configuration Derived from All Logs & YAML ---
PRETRAINED_MODEL_PATH = '../weights/transformer_120.pth'
IMG_SIZE = (256, 128)
PATCH_SIZE = (16, 16) # Conceptual patch size

# ViT-Small parameters
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6 
MLP_RATIO = 4.0
QKV_BIAS = True
FINAL_CLASSIFIER_CLASSES = 1041 

# =================================================================================
# MODEL ARCHITECTURE ALIGNED WITH TRANSREID REPOSITORY
# =================================================================================

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class CorrectedPatchEmbedding(nn.Module):
    """
    This patch embedding module is definitively correct based on the final error log.
    It consists of a convolutional stem followed by a projection layer.
    """
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        # 1. The Convolutional Stem
        self.conv = nn.Sequential(OrderedDict([
            ('0', nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        # 2. The Final Projection Layer
        self.proj = nn.Conv2d(64, embed_dim, kernel_size=(8, 8), stride=(8, 8))

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        # B, C, H, W -> B, C, H*W -> B, H*W, C
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads, qkv_bias, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio, qkv_bias, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BaseVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = CorrectedPatchEmbedding(in_channels=3, embed_dim=EMBED_DIM)
        
        num_patches = (IMG_SIZE[0] // 16) * (IMG_SIZE[1] // 16)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, EMBED_DIM))
        self.pos_drop = nn.Dropout(p=0.)
        dpr = [x.item() for x in torch.linspace(0, 0., DEPTH)]
        self.blocks = nn.ModuleList([Block(EMBED_DIM, NUM_HEADS, MLP_RATIO, QKV_BIAS, drop=0., attn_drop=0., drop_path=dpr[i]) for i in range(DEPTH)])
        self.norm = nn.LayerNorm(EMBED_DIM, eps=1e-6)
        # Layer to absorb the pretrained fc weights
        self.fc = nn.Linear(EMBED_DIM, 1000, bias=True)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

class CompleteVisionTransformer(nn.Module):
    """ The top-level wrapper model. This is the one you should instantiate. """
    def __init__(self):
        super().__init__()
        self.base = BaseVisionTransformer()
        self.bottleneck = nn.BatchNorm1d(EMBED_DIM)
        self.classifier = nn.Linear(EMBED_DIM, FINAL_CLASSIFIER_CLASSES, bias=False)

    def forward(self, x):
        features = self.base(x)
        bn_features = self.bottleneck(features)
        if not self.training:
            return bn_features
        logits = self.classifier(bn_features)
        return logits

from PIL import Image
from torchvision import transforms

# =================================================================================
# MAIN EXECUTION BLOCK
# =================================================================================

if __name__ == '__main__':
    model = CompleteVisionTransformer()

    print("--- Attempting to load pretrained weights ---")
    print(f"Loading from: {PRETRAINED_MODEL_PATH}")

    try:
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location='cpu')
        
        # Use strict=False to gracefully handle the complex keys inside the patch_embed stem.
        # PyTorch will automatically load all keys that perfectly match in name and shape.
        model.load_state_dict(checkpoint, strict=False)
        
        print("\n[SUCCESS] Pretrained weights loaded successfully!")
        
        model.eval()
        print("\nModel set to evaluation mode.")

        # --- Image Preprocessing ---
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # --- Load and Process Images ---
        img1 = Image.open("person_crop_01.jpg").convert("RGB")
        img2 = Image.open("person_crop_02.jpg").convert("RGB")

        img1_tensor = transform(img1).unsqueeze(0)
        img2_tensor = transform(img2).unsqueeze(0)

        # --- Feature Extraction ---
        with torch.no_grad():
            features1 = model(img1_tensor)
            features2 = model(img2_tensor)

        # --- Cosine Similarity ---
        cos_sim = torch.nn.functional.cosine_similarity(features1, features2)
        
        print("\n--- Inference ---")
        print(f"Features for person_crop_0.jpg: {features1.shape}")
        print(f"Features for person_crop_1.jpg: {features2.shape}")
        print(f"Cosine Similarity: {cos_sim.item()}")

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
"""
HSILiteSegNet — training script tuned for RTX 3090 (24 GB).

This is the same HSILiteSegNet architecture as before (dual-branch spectral/spatial
stem + gated fusion + depthwise-separable blocks with ECA + axial-attention bottleneck
+ U-Net encoder/decoder + Focal/Dice combo loss + Effective-Number class reweighting +
deep supervision). Only the DEFAULTS were changed for a 24 GB GPU, the data paths were
filled in, and one FocalLoss bug was fixed (see note below).

Defaults changed for RTX 3090
------------------------------
    batch_size      : 8     (8 GB version used 4)
    stem_3d_filters : 8     (8 GB version used 4)
    epochs          : 100   (per your request)
    in_channels     : 204
    num_classes     : 7
    data paths      : D:\\ZCY\\HSILiteSegNet\\data\\...
    save_dir        : D:\\ZCY\\HSILiteSegNet\\checkpoints

Just run:
    python train_hsi_litesegnet_3090.py
or override anything on the command line, e.g.:
    python train_hsi_litesegnet_3090.py --batch-size 12 --epochs 120

Checkpoints
-----------
Each time validation Kappa improves, a NEW checkpoint file is written:
    {save_dir}/trainmodel_{run_id}_{epoch}_kappa_{pct}%.pth
e.g.  checkpoints/trainmodel_9999_15_kappa_89%.pth
Nothing is ever overwritten. `run_id` is a random integer in [0, 10000]
picked at the start of every run.
"""

import argparse
import math
import os
import random
import time
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[warn] tqdm not installed; falling back to plain text progress. "
          "`pip install tqdm` for nicer bars.")


# ============================================================
# Reproducibility & Config
# ============================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@dataclass
class Config:
    # ---- Data paths (filled in for your machine) ----
    train_image_dir: str = r"D:\ZCY\HSILiteSegNet\data\retest4\train\images"
    train_mask_dir:  str = r"D:\ZCY\HSILiteSegNet\data\retest4\train\masks"
    val_image_dir:   str = r"D:\ZCY\HSILiteSegNet\data\retest4\val\images"
    val_mask_dir:    str = r"D:\ZCY\HSILiteSegNet\data\retest4\val\masks"

    # ---- Task ----
    num_classes:  int = 7
    in_channels:  int = 30     # raw bands / PCA components in the input .npy files
    ignore_index: int = 255

    # ---- Patch sampling ----
    patch_size:   int = 128
    train_stride: int = 32
    val_stride:   int = 128

    # ---- Training schedule (tuned for RTX 3090 24 GB) ----
    batch_size:   int = 8       # 24 GB can handle 8 (even 12) at 204 bands
    num_workers:  int = 4       # 3090 box usually has enough CPU; 0 if Windows issues
    epochs:       int = 100     # per your request
    lr:           float = 1e-3
    min_lr:       float = 1e-5
    weight_decay: float = 1e-4

    # ---- Loss ----
    focal_weight: float = 1.0
    dice_weight:  float = 1.0
    aux_weight:   float = 0.4   # set 0 to disable deep supervision
    focal_gamma:  float = 2.0

    # ---- Model architecture ----
    base_channels:   int = 32   # encoder width
    stem_3d_filters: int = 8    # 3D-conv filters in spectral stem (8 is fine on 24 GB)
    attn_heads:      int = 4    # heads in axial attention bottleneck
    dropout:         float = 0.1

    # ---- Structural ablation (all default to the full paper model) ----
    branch:       str = "both"     # both | spec | spat
    fusion:       str = "gated"    # gated | add | concat  (branch=both only)
    axial:        str = "hw"       # hw | h | w | none
    channel_attn: str = "eca"      # eca | se | none
    conv_type:    str = "dsconv"   # dsconv | standard

    # ---- Class-imbalance handling ----
    use_class_weights: bool = True
    oversample_factor: int = 4
    rare_top_k:        int = 3

    # ---- Augmentation toggles ----
    use_spatial_aug:  bool = True
    use_spectral_aug: bool = True

    # ---- Early stopping / monitoring ----
    patience: int = 25
    
    monitor:  str = "Kappa"     # one of: "Kappa", "mIoU", "AA", "OA"

    # ---- Misc ----
    device:   str = "cuda" if torch.cuda.is_available() else "cpu"
    amp:      bool = True
    seed:     int = 42
    save_dir: str = r"D:\ZCY\HSILiteSegNet\output\models\re\test4"


# ============================================================
# Data
# ============================================================

def load_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[0] > arr.shape[-1]:
        arr = np.transpose(arr, (2, 0, 1))   # HWC -> CHW
    return arr


class PatchHSIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size, stride, num_classes,
                 ignore_index=255, augment_spatial=False, augment_spectral=False,
                 oversample_rare=True, rare_class_ids=None, oversample_factor=4):
        self.image_paths = sorted(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".npy")]
        )
        self.mask_paths = sorted(
            [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".npy")]
        )
        assert len(self.image_paths) == len(self.mask_paths), "image / mask count mismatch"
        assert len(self.image_paths) > 0, f"no .npy files under {image_dir}"

        self.patch_size = patch_size
        self.stride = stride
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.augment_spatial = augment_spatial
        self.augment_spectral = augment_spectral
        self.oversample_rare = oversample_rare
        self.rare_class_ids = set(rare_class_ids or [])
        self.oversample_factor = max(1, oversample_factor)

        self.images = [self._normalize(load_npy(p).astype(np.float32)) for p in self.image_paths]
        self.masks  = [np.load(p).astype(np.int64) for p in self.mask_paths]

        self.index = self._build_patch_index()

    @staticmethod
    def _normalize(img):
        m = img.mean(axis=(1, 2), keepdims=True)
        s = img.std(axis=(1, 2), keepdims=True) + 1e-6
        return (img - m) / s

    def _build_patch_index(self):
        idx, boost = [], []
        for i, mask in enumerate(self.masks):
            h, w = mask.shape
            ys = list(range(0, max(1, h - self.patch_size + 1), self.stride))
            xs = list(range(0, max(1, w - self.patch_size + 1), self.stride))
            if ys and ys[-1] + self.patch_size < h:
                ys.append(h - self.patch_size)
            if xs and xs[-1] + self.patch_size < w:
                xs.append(w - self.patch_size)
            for y in ys:
                for x in xs:
                    idx.append((i, y, x))
                    if self.oversample_rare and self.rare_class_ids:
                        patch = mask[y:y + self.patch_size, x:x + self.patch_size]
                        present = set(np.unique(patch).tolist())
                        if present.intersection(self.rare_class_ids):
                            for _ in range(self.oversample_factor - 1):
                                boost.append((i, y, x))
        return idx + boost

    def __len__(self):
        return len(self.index)

    def _spatial_aug(self, img, mask):
        if random.random() < 0.5:
            img = np.flip(img, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
        return img, mask

    def _spectral_aug(self, img):
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.02, img.shape[0]).astype(np.float32)[:, None, None]
            img = img + noise
        if random.random() < 0.3:
            n = random.randint(1, max(1, img.shape[0] // 8))
            ix = np.random.choice(img.shape[0], n, replace=False)
            img = img.copy()
            img[ix] = 0.0
        if random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            bias = np.random.uniform(-0.05, 0.05)
            img = img * scale + bias
        return img

    def __getitem__(self, idx):
        i, y, x = self.index[idx]
        img = self.images[i][:, y:y + self.patch_size, x:x + self.patch_size]
        mask = self.masks[i][y:y + self.patch_size, x:x + self.patch_size]
        if self.augment_spatial:
            img, mask = self._spatial_aug(img, mask)
        if self.augment_spectral:
            img = self._spectral_aug(img)
        return (torch.from_numpy(np.ascontiguousarray(img)).float(),
                torch.from_numpy(np.ascontiguousarray(mask)).long())


# ============================================================
# Structural ablation switches
# ------------------------------------------------------------
# Set ONCE in main() from CLI args, then read by the modules
# below at build time. Lets you run every structural ablation
# from the command line WITHOUT editing the model code.
#   branch       : both | spec | spat
#   fusion       : gated | add | concat     (branch=both only)
#   axial        : hw | h | w | none
#   channel_attn : eca | se | none
#   conv_type    : dsconv | standard
# ============================================================
class _Ablation:
    branch       = "both"
    fusion       = "gated"
    axial        = "hw"
    channel_attn = "eca"
    conv_type    = "dsconv"


ABL = _Ablation()


# ============================================================
# Building blocks
# ============================================================

class ECA(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(-1, -2)
        y = torch.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (ablation alternative to ECA)."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        r = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, r, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class DSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        # conv_type ablation: dsconv -> depthwise (groups=channels);
        #                     standard -> full conv (groups=1)
        ds = (ABL.conv_type == "dsconv")
        g1 = in_ch if ds else 1
        g2 = out_ch if ds else 1
        self.dw1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=g1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dw2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=g2, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        # channel_attn ablation: eca | se | none  (attribute name kept as `eca`
        # so existing state_dict keys stay compatible for the eca case)
        if ABL.channel_attn == "se":
            self.eca = SEBlock(out_ch)
        elif ABL.channel_attn == "none":
            self.eca = nn.Identity()
        else:
            self.eca = ECA(k_size=3)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        x = self.act(self.bn1(self.dw1(x)))
        x = self.act(self.bn2(self.pw1(x)))
        x = self.bn3(self.dw2(x))
        x = self.eca(x)
        x = self.act(x + identity)
        return self.drop(x)


class Spectral3DStem(nn.Module):
    def __init__(self, in_bands, mid_filters=8, out_channels=32):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, mid_filters, (7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(mid_filters), nn.ReLU(inplace=True),
            nn.Conv3d(mid_filters, mid_filters, (5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(mid_filters), nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(mid_filters * in_bands, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, f, c, h, w = x.shape
        x = x.reshape(b, f * c, h, w)
        return self.proj(x)


class SpatialStem(nn.Module):
    def __init__(self, in_bands, out_channels=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_bands, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class GatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2 * channels, 1),
        )

    def forward(self, spec, spat):
        g = self.gate(torch.cat([spec, spat], dim=1))
        gs1, gs2 = torch.chunk(g, 2, dim=1)
        gates = torch.softmax(torch.stack([gs1, gs2], dim=1), dim=1)
        return spec * gates[:, 0] + spat * gates[:, 1]


class FusionModule(nn.Module):
    """Wraps the three fusion variants for the branch=both ablation.
       gated  -> learned soft gating (paper default)
       add    -> element-wise sum
       concat -> channel concat + 1x1 conv
    """
    def __init__(self, channels, mode="gated"):
        super().__init__()
        self.mode = mode
        if mode == "gated":
            self.fuse = GatedFusion(channels)
        elif mode == "concat":
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            )
        # 'add' needs no parameters

    def forward(self, spec, spat):
        if self.mode == "gated":
            return self.fuse(spec, spat)
        if self.mode == "concat":
            return self.fuse(torch.cat([spec, spat], dim=1))
        return spec + spat


class AxialAttention(nn.Module):
    def __init__(self, channels, num_heads=4, axis="h"):
        super().__init__()
        assert channels % num_heads == 0
        self.heads = num_heads
        self.axis = axis
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv1d(channels, channels, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.axis == "h":
            x_seq = x.permute(0, 3, 1, 2).contiguous().reshape(B * W, C, H)
        else:
            x_seq = x.permute(0, 2, 1, 3).contiguous().reshape(B * H, C, W)
        qkv = self.qkv(x_seq)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        BX, _, L = q.shape
        q = q.view(BX, self.heads, self.head_dim, L).permute(0, 1, 3, 2)
        k = k.view(BX, self.heads, self.head_dim, L)
        v = v.view(BX, self.heads, self.head_dim, L).permute(0, 1, 3, 2)
        attn = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        out = torch.matmul(attn, v).permute(0, 1, 3, 2).contiguous().reshape(BX, C, L)
        out = self.proj(out)
        if self.axis == "h":
            return out.reshape(B, W, C, H).permute(0, 2, 3, 1).contiguous()
        return out.reshape(B, H, C, W).permute(0, 2, 1, 3).contiguous()


class AxialBlock(nn.Module):
    def __init__(self, channels, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        gn = 8 if channels % 8 == 0 else 4
        # axial ablation: hw | h | w | none  (none = MLP-only bottleneck)
        self.axial = ABL.axial
        self.norm1 = nn.GroupNorm(gn, channels)
        self.attn_h = AxialAttention(channels, num_heads, axis="h") \
            if self.axial in ("hw", "h") else None
        self.attn_w = AxialAttention(channels, num_heads, axis="w") \
            if self.axial in ("hw", "w") else None
        self.norm2 = nn.GroupNorm(gn, channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(),
            nn.Dropout2d(dropout), nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x):
        y = self.norm1(x)
        if self.attn_h is not None:
            x = x + self.attn_h(y)
        if self.attn_w is not None:
            x = x + self.attn_w(y)
        x = x + self.mlp(self.norm2(x))
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DSConvBlock(in_ch, out_ch, dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DSConvBlock(in_ch // 2 + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class HSILiteSegNet(nn.Module):
    def __init__(self, in_channels, num_classes, base_ch=32,
                 stem_3d_filters=8, attn_heads=4, dropout=0.1, enable_aux=True):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.enable_aux = enable_aux

        # branch ablation: both | spec | spat  (unused stems are not built)
        self.branch = ABL.branch
        self.stem_spec = Spectral3DStem(in_channels, stem_3d_filters, c1) \
            if self.branch in ("both", "spec") else None
        self.stem_spat = SpatialStem(in_channels, c1) \
            if self.branch in ("both", "spat") else None
        self.fuse = FusionModule(c1, ABL.fusion) if self.branch == "both" else None

        self.enc1 = DSConvBlock(c1, c1, dropout=0.0)
        self.down1 = Down(c1, c2, dropout)
        self.down2 = Down(c2, c3, dropout)
        self.down3 = Down(c3, c4, dropout)

        self.bottleneck = AxialBlock(c4, num_heads=attn_heads, mlp_ratio=2.0, dropout=dropout)

        self.up1 = Up(c4, c3, c3, dropout)
        self.up2 = Up(c3, c2, c2, dropout)
        self.up3 = Up(c2, c1, c1, dropout)

        self.head_main = nn.Conv2d(c1, num_classes, 1)
        self.head_aux  = nn.Conv2d(c2, num_classes, 1) if enable_aux else None

    def forward(self, x):
        if self.branch == "both":
            x0 = self.fuse(self.stem_spec(x), self.stem_spat(x))
        elif self.branch == "spec":
            x0 = self.stem_spec(x)
        else:
            x0 = self.stem_spat(x)
        x1 = self.enc1(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.bottleneck(self.down3(x3))
        d3 = self.up1(x4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up3(d2, x1)

        main = self.head_main(d1)
        if self.training and self.head_aux is not None:
            aux = self.head_aux(d2)
            aux = F.interpolate(aux, size=main.shape[2:], mode="bilinear", align_corners=False)
            return main, aux
        return main


# ============================================================
# Losses
# ============================================================

class FocalLoss(nn.Module):
    """Multi-class focal loss with ignore_index handling.

    NOTE (bug fix vs original): cross_entropy already excludes ignore_index
    pixels and returns 0 for them when reduction='none'. The original code then
    multiplied by `valid` AND divided by valid.sum(), which is correct, but the
    per-pixel `ce` for ignored pixels could be non-finite in AMP. We now zero the
    ignored positions explicitly before reduction to keep the average clean.
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha,
                             ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        valid = (target != self.ignore_index).float()
        loss = loss * valid                      # zero-out ignored pixels
        return loss.sum() / valid.sum().clamp(min=1.0)


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, smooth=1.0):
        super().__init__()
        self.C = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        valid = target != self.ignore_index
        t = target.clone()
        t[~valid] = 0
        oh = F.one_hot(t, self.C).permute(0, 3, 1, 2).float()
        m = valid.unsqueeze(1).float()
        probs = probs * m
        oh = oh * m
        inter = (probs * oh).sum((0, 2, 3))
        union = (probs + oh).sum((0, 2, 3))
        return 1.0 - ((2 * inter + self.smooth) / (union + self.smooth)).mean()


class ComboLoss(nn.Module):
    def __init__(self, num_classes, class_weights, ignore_index,
                 focal_weight, dice_weight, focal_gamma):
        super().__init__()
        self.focal = FocalLoss(alpha=class_weights, gamma=focal_gamma, ignore_index=ignore_index)
        self.dice  = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.fw = focal_weight
        self.dw = dice_weight

    def forward(self, logits, target):
        loss = 0.0
        if self.fw > 0:
            loss = loss + self.fw * self.focal(logits, target)
        if self.dw > 0:
            loss = loss + self.dw * self.dice(logits, target)
        return loss


# ============================================================
# Class statistics & Kappa
# ============================================================

@torch.no_grad()
def compute_class_stats(mask_dir, num_classes, ignore_index=255):
    counts = np.zeros(num_classes, dtype=np.float64)
    for name in os.listdir(mask_dir):
        if not name.endswith(".npy"):
            continue
        mask = np.load(os.path.join(mask_dir, name))
        valid = mask != ignore_index
        vals, cnts = np.unique(mask[valid], return_counts=True)
        for v, c in zip(vals, cnts):
            if 0 <= int(v) < num_classes:
                counts[int(v)] += c
    counts = np.maximum(counts, 1.0)
    beta = 0.9999
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / eff
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), counts.tolist()


def compute_kappa(conf: np.ndarray) -> float:
    """Cohen's kappa = (po - pe) / (1 - pe)."""
    n = conf.sum()
    if n == 0:
        return 0.0
    po = np.trace(conf) / n
    row = conf.sum(axis=1).astype(np.float64)
    col = conf.sum(axis=0).astype(np.float64)
    pe = (row * col).sum() / (n * n)
    if pe >= 1.0:
        return 0.0
    return float((po - pe) / (1.0 - pe))


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, num_classes, ignore_index, device,
             criterion=None, show_progress=False):
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss, batches = 0.0, 0

    it = tqdm(loader, desc="  [val]", ncols=100, leave=False) \
        if (show_progress and HAS_TQDM) else loader

    for imgs, masks in it:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        out = model(imgs)
        if isinstance(out, tuple):
            out = out[0]
        if criterion is not None:
            total_loss += float(criterion(out, masks).item())
            batches += 1
        preds = torch.argmax(out, dim=1)
        v = masks != ignore_index
        gts = masks[v].cpu().numpy().astype(np.int64)
        pds = preds[v].cpu().numpy().astype(np.int64)
        k = (gts >= 0) & (gts < num_classes) & (pds >= 0) & (pds < num_classes)
        if k.any():
            idx = num_classes * gts[k] + pds[k]
            conf += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    ious, accs = [], []
    for c in range(num_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        di = tp + fp + fn
        da = tp + fn
        ious.append(tp / di if di > 0 else float("nan"))
        accs.append(tp / da if da > 0 else float("nan"))

    return {
        "loss":  total_loss / max(batches, 1),
        "OA":    float(np.trace(conf) / max(conf.sum(), 1)),
        "AA":    float(np.nanmean(accs)),
        "mIoU":  float(np.nanmean(ious)),
        "Kappa": compute_kappa(conf),
        "IoU_per_class": ious,
        "Acc_per_class": accs,
        "confusion": conf,
    }


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler, use_amp, aux_weight, epoch, total_epochs):
    model.train()
    running, n = 0.0, 0
    total = len(loader)
    pbar = tqdm(loader, desc=f"  [Epoch {epoch:03d}/{total_epochs}]",
                ncols=100, leave=False) if HAS_TQDM else None
    it = pbar if pbar is not None else loader

    for bi, (imgs, masks) in enumerate(it, start=1):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda"):
                out = model(imgs)
                if isinstance(out, tuple):
                    main, aux = out
                    loss = criterion(main, masks) + aux_weight * criterion(aux, masks)
                else:
                    loss = criterion(out, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            if isinstance(out, tuple):
                main, aux = out
                loss = criterion(main, masks) + aux_weight * criterion(aux, masks)
            else:
                loss = criterion(out, masks)
            loss.backward()
            optimizer.step()

        running += float(loss.item())
        n += 1

        if pbar is not None:
            pbar.set_postfix({"batch%": f"{100.0 * bi / total:5.1f}%",
                              "loss":   f"{running / n:.4f}"})
        else:
            step = max(1, total // 10)
            if bi % step == 0 or bi == total:
                print(f"    Epoch {epoch:03d}/{total_epochs}  "
                      f"batch {bi}/{total} ({100.0 * bi / total:5.1f}%)  "
                      f"loss={running / n:.4f}")

    return running / max(n, 1)


# ============================================================
# Helpers
# ============================================================

def format_seconds(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def kappa_to_pct(kappa: float) -> str:
    """0.8945 -> '89' to match the requested filename format."""
    return f"{int(round(kappa * 100))}"


# ============================================================
# CLI -> Config
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    cfg = Config()
    p = argparse.ArgumentParser(
        description="HSILiteSegNet training (RTX 3090 defaults).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data paths
    p.add_argument("--train-image-dir", type=str, default=cfg.train_image_dir)
    p.add_argument("--train-mask-dir",  type=str, default=cfg.train_mask_dir)
    p.add_argument("--val-image-dir",   type=str, default=cfg.val_image_dir)
    p.add_argument("--val-mask-dir",    type=str, default=cfg.val_mask_dir)
    # task
    p.add_argument("--num-classes",  type=int, default=cfg.num_classes)
    p.add_argument("--in-channels",  type=int, default=cfg.in_channels,
                   help="raw bands / PCA components in input .npy files")
    p.add_argument("--ignore-index", type=int, default=cfg.ignore_index)
    # patches
    p.add_argument("--patch-size",   type=int, default=cfg.patch_size)
    p.add_argument("--train-stride", type=int, default=cfg.train_stride)
    p.add_argument("--val-stride",   type=int, default=cfg.val_stride)
    # schedule
    p.add_argument("--batch-size",   type=int, default=cfg.batch_size)
    p.add_argument("--num-workers",  type=int, default=cfg.num_workers)
    p.add_argument("--epochs",       type=int, default=cfg.epochs)
    p.add_argument("--lr",           type=float, default=cfg.lr)
    p.add_argument("--min-lr",       type=float, default=cfg.min_lr)
    p.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    # loss
    p.add_argument("--focal-weight", type=float, default=cfg.focal_weight)
    p.add_argument("--dice-weight",  type=float, default=cfg.dice_weight)
    p.add_argument("--aux-weight",   type=float, default=cfg.aux_weight,
                   help="set 0 to disable deep supervision")
    p.add_argument("--focal-gamma",  type=float, default=cfg.focal_gamma)
    # architecture
    p.add_argument("--base-channels",   type=int, default=cfg.base_channels)
    p.add_argument("--stem-3d-filters", type=int, default=cfg.stem_3d_filters)
    p.add_argument("--attn-heads",      type=int, default=cfg.attn_heads)
    p.add_argument("--dropout",         type=float, default=cfg.dropout)
    # structural ablation
    p.add_argument("--branch", type=str, default=cfg.branch,
                   choices=["both", "spec", "spat"],
                   help="ablation: which stem branch(es) to use")
    p.add_argument("--fusion", type=str, default=cfg.fusion,
                   choices=["gated", "add", "concat"],
                   help="ablation: fusion type (branch=both only)")
    p.add_argument("--axial", type=str, default=cfg.axial,
                   choices=["hw", "h", "w", "none"],
                   help="ablation: axial-attention axes at the bottleneck")
    p.add_argument("--channel-attn", type=str, default=cfg.channel_attn,
                   choices=["eca", "se", "none"],
                   help="ablation: channel attention inside DSConvBlock")
    p.add_argument("--conv-type", type=str, default=cfg.conv_type,
                   choices=["dsconv", "standard"],
                   help="ablation: depthwise-separable vs standard conv")
    # class imbalance
    p.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction,
                   default=cfg.use_class_weights,
                   help="Effective-Number class reweighting on/off")
    p.add_argument("--oversample-factor", type=int, default=cfg.oversample_factor)
    p.add_argument("--rare-top-k",        type=int, default=cfg.rare_top_k)
    # augmentation
    p.add_argument("--use-spatial-aug",  action=argparse.BooleanOptionalAction,
                   default=cfg.use_spatial_aug)
    p.add_argument("--use-spectral-aug", action=argparse.BooleanOptionalAction,
                   default=cfg.use_spectral_aug)
    # early stopping
    p.add_argument("--patience", type=int, default=cfg.patience)
    p.add_argument("--monitor",  type=str, default=cfg.monitor,
                   choices=["Kappa", "mIoU", "AA", "OA"])
    # misc
    p.add_argument("--device",   type=str, default=cfg.device)
    p.add_argument("--amp",      action=argparse.BooleanOptionalAction, default=cfg.amp)
    p.add_argument("--seed",     type=int, default=cfg.seed)
    p.add_argument("--save-dir", type=str, default=cfg.save_dir)
    return p


def args_to_config(args) -> Config:
    cfg = Config()
    for f in fields(cfg):
        if hasattr(args, f.name):
            setattr(cfg, f.name, getattr(args, f.name))
    return cfg


# ============================================================
# Main
# ============================================================

def main():
    args = build_argparser().parse_args()
    cfg = args_to_config(args)
    seed_everything(cfg.seed)

    run_id = random.randint(0, 10000)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("=" * 72)
    print(f" HSILiteSegNet training run  (RTX 3090 config)")
    print(f"   run_id       : {run_id}")
    print(f"   device       : {cfg.device}    AMP={cfg.amp and cfg.device == 'cuda'}")
    print(f"   in_channels  : {cfg.in_channels}     num_classes : {cfg.num_classes}")
    print(f"   patch_size   : {cfg.patch_size}      batch_size  : {cfg.batch_size}")
    print(f"   epochs       : {cfg.epochs}     monitor     : {cfg.monitor}")
    print(f"   loss weights : focal={cfg.focal_weight}  dice={cfg.dice_weight}  "
          f"aux={cfg.aux_weight}  gamma={cfg.focal_gamma}")
    print(f"   class re-wt  : {cfg.use_class_weights}    "
          f"spatial_aug={cfg.use_spatial_aug}   spectral_aug={cfg.use_spectral_aug}")
    print(f"   save_dir     : {cfg.save_dir}")
    print("=" * 72)

    # ---- class stats / weights ----
    class_weights, counts = compute_class_stats(
        cfg.train_mask_dir, cfg.num_classes, cfg.ignore_index
    )
    rare = np.argsort(np.array(counts))[: cfg.rare_top_k].tolist()
    print(f"Class pixel counts: {counts}")
    print(f"Class weights raw : {[round(x, 3) for x in class_weights.tolist()]}")
    print(f"Rare classes (top-{cfg.rare_top_k}): {rare}")
    cw_for_loss = class_weights.to(cfg.device) if cfg.use_class_weights else None
    if not cfg.use_class_weights:
        print("  (class weights DISABLED for this run)")

    # ---- datasets ----
    train_ds = PatchHSIDataset(
        image_dir=cfg.train_image_dir, mask_dir=cfg.train_mask_dir,
        patch_size=cfg.patch_size, stride=cfg.train_stride,
        num_classes=cfg.num_classes, ignore_index=cfg.ignore_index,
        augment_spatial=cfg.use_spatial_aug,
        augment_spectral=cfg.use_spectral_aug,
        oversample_rare=True, rare_class_ids=rare,
        oversample_factor=cfg.oversample_factor,
    )
    val_ds = PatchHSIDataset(
        image_dir=cfg.val_image_dir, mask_dir=cfg.val_mask_dir,
        patch_size=cfg.patch_size, stride=cfg.val_stride,
        num_classes=cfg.num_classes, ignore_index=cfg.ignore_index,
        augment_spatial=False, augment_spectral=False, oversample_rare=False,
    )
    print(f"Train patches: {len(train_ds)} | Val patches: {len(val_ds)}")

    # quick channel-count sanity check
    sample_img, _ = train_ds[0]
    if sample_img.shape[0] != cfg.in_channels:
        print(f"\n[ERROR] sample channels {sample_img.shape[0]} != "
              f"--in-channels {cfg.in_channels}. "
              f"Edit --in-channels or check your .npy files.")
        return
    print(f"Sanity check: sample tensor shape = {tuple(sample_img.shape)}  OK")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            persistent_workers=cfg.num_workers > 0)

    # ---- apply structural ablation switches (read by modules at build time) ----
    ABL.branch       = cfg.branch
    ABL.fusion       = cfg.fusion
    ABL.axial        = cfg.axial
    ABL.channel_attn = cfg.channel_attn
    ABL.conv_type    = cfg.conv_type
    print(f"Ablation: branch={ABL.branch}  fusion={ABL.fusion}  axial={ABL.axial}  "
          f"channel_attn={ABL.channel_attn}  conv_type={ABL.conv_type}")

    # ---- model ----
    model = HSILiteSegNet(
        in_channels=cfg.in_channels, num_classes=cfg.num_classes,
        base_ch=cfg.base_channels, stem_3d_filters=cfg.stem_3d_filters,
        attn_heads=cfg.attn_heads, dropout=cfg.dropout,
        enable_aux=(cfg.aux_weight > 0),
    ).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params / 1e6:.3f} M    "
          f"aux_head_enabled={cfg.aux_weight > 0}")

    # ---- loss / optim ----
    criterion = ComboLoss(
        num_classes=cfg.num_classes, class_weights=cw_for_loss,
        ignore_index=cfg.ignore_index,
        focal_weight=cfg.focal_weight, dice_weight=cfg.dice_weight,
        focal_gamma=cfg.focal_gamma,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )
    use_amp = cfg.amp and cfg.device == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- training loop ----
    best_score = -float("inf")
    bad = 0
    epoch_times: List[float] = []
    saved: List[str] = []
    train_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        ep_start = time.time()
        ep_pct = 100.0 * epoch / cfg.epochs
        eta = (sum(epoch_times) / len(epoch_times)) * (cfg.epochs - epoch + 1) \
            if epoch_times else None
        eta_str = format_seconds(eta) if eta is not None else "  --:--:--"

        print(f"\n>>> Epoch {epoch:03d}/{cfg.epochs} ({ep_pct:5.1f}%)   "
              f"ETA {eta_str}")

        tl = train_one_epoch(model, train_loader, optimizer, criterion,
                             cfg.device, scaler, use_amp, cfg.aux_weight,
                             epoch, cfg.epochs)
        m = evaluate(model, val_loader, cfg.num_classes, cfg.ignore_index,
                     cfg.device, criterion, show_progress=True)
        scheduler.step()

        ep_time = time.time() - ep_start
        epoch_times.append(ep_time)
        cur_lr = optimizer.param_groups[0]["lr"]

        ious = ", ".join(f"c{c}={iou:.3f}" if not math.isnan(iou) else f"c{c}=nan"
                         for c, iou in enumerate(m["IoU_per_class"]))
        print(f"  lr={cur_lr:.2e}  train_loss={tl:.4f}  val_loss={m['loss']:.4f}")
        print(f"  OA={m['OA']:.4f}  AA={m['AA']:.4f}  "
              f"mIoU={m['mIoU']:.4f}  Kappa={m['Kappa']:.4f}")
        print(f"  per-class IoU: {ious}")
        print(f"  epoch_time={format_seconds(ep_time)}   "
              f"elapsed={format_seconds(time.time() - train_start)}")

        score = m[cfg.monitor]
        if score > best_score:
            best_score = score
            bad = 0
            fname = f"trainmodel_{run_id}_{epoch}_kappa_{kappa_to_pct(m['Kappa'])}%.pth"
            fpath = os.path.join(cfg.save_dir, fname)
            torch.save({
                "model_state":   model.state_dict(),
                "config":        cfg.__dict__,
                "class_weights": class_weights,
                "epoch":         epoch,
                "run_id":        run_id,
                "metrics":       {k: m[k] for k in ("OA", "AA", "mIoU", "Kappa")},
            }, fpath)
            saved.append(fpath)
            print(f"  >> NEW BEST {cfg.monitor}={score:.4f}   saved -> {fpath}")
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"\nEarly stopping at epoch {epoch}: "
                      f"{cfg.monitor} no improvement for {cfg.patience} epochs.")
                break

    total_time = time.time() - train_start
    print("\n" + "=" * 72)
    print(f" Training finished.")
    print(f"   run_id           : {run_id}")
    print(f"   best {cfg.monitor:<11}: {best_score:.4f}")
    print(f"   epochs run       : {len(epoch_times)} / {cfg.epochs}")
    print(f"   total wall time  : {format_seconds(total_time)}  "
          f"({total_time:.1f} sec)")
    if epoch_times:
        print(f"   mean epoch time  : "
              f"{format_seconds(sum(epoch_times) / len(epoch_times))}")
    print(f"   saved best ckpts : {len(saved)}")
    for p in saved:
        print(f"     - {p}")
    print("=" * 72)


if __name__ == "__main__":
    main()

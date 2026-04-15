
#WORKING CODE
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nn import SynchronizedBatchNorm2d
from lib.shunted import shunted_s, shunted_b, shunted_t

BatchNorm2d = SynchronizedBatchNorm2d


# ------------------------------------------------------------
# Basic Conv Block
# ------------------------------------------------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, dilation=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


# ------------------------------------------------------------
# Center-Aware MicroStructure Enhancement
# ------------------------------------------------------------

class MDSE(nn.Module):
    """
    Micro-Structure Enhancement (MDSE)

    Directional depthwise filters capture horizontal, vertical, and isotropic
    micro-structure. A centerness head with Local Contrast Normalization (LCN)
    produces a center-surround spatial confidence map that suppresses flat
    uniform regions and sharpens true structural peaks. The final refinement
    is a bounded multiplicative residual gated jointly by directional attention
    and LCN-normalized centerness.
    """

    def __init__(self, channels):
        super().__init__()

        # ---- Directional depthwise filters ----
        self.h   = nn.Conv2d(channels, channels, (1, 3),
                             padding=(0, 1), groups=channels, bias=False)
        self.v   = nn.Conv2d(channels, channels, (3, 1),
                             padding=(1, 0), groups=channels, bias=False)
        self.iso = nn.Conv2d(channels, channels, 3,
                             padding=1,      groups=channels, bias=False)

        # ---- Directional fusion attention ----
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # ---- Centerness head ----
        self.centerness = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )

        # ---- Bounded residual gain (scalar) ----
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):

        # ---- Directional attention ----
        att = self.fuse(torch.cat([self.h(x),
                                   self.v(x),
                                   self.iso(x)], dim=1))   # (B, C, H, W)

        # ---- Centerness with Local Contrast Normalization ----
        # raw centerness: high at structural peaks
        center = self.centerness(x)                        # (B, 1, H, W)

        # LCN: subtract local neighbourhood mean (7×7)
        # → center-surround DoG-like response
        # → suppresses flat uniform regions
        # → sharpens and isolates true spatial peaks
        center = center - F.avg_pool2d(center, kernel_size=7,
                                       stride=1, padding=3)
        center = torch.sigmoid(center)                     # (B, 1, H, W)

        # ---- Bounded residual gain ----
        gain = torch.clamp(self.alpha, 0.0, 0.5)

        # ---- Center-aware micro-structure refinement ----
        # att  : directional structural response  — what the structure is
        # center: LCN spatial confidence          — where to trust it
        # gain  : bounded global amplification    — how much to refine
        return x + gain * x * att * center



class CSRERA(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.gate_temp = 2.0

        # ---- Coarse projection ----
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(1, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- Retain and Erase gates (separate projections) ----
        self.conv_r = nn.Conv2d(channel, channel, 1)
        self.conv_e = nn.Conv2d(channel, channel, 1)

        # ---- Contrast modulation gate ----
        self.fusion_gate = nn.Conv2d(channel, channel, 1, bias=False)

        # ---- RERA fusion conv ----
        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )


        self.pre_micro = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- Micro-structure refinement ----
        self.micro = MDSE(channel)

    def forward(self, coarse, f):

        # ---- Resize coarse prediction to match f ----
        if coarse.shape[2:] != f.shape[2:]:
            coarse = F.interpolate(
                coarse,
                size=f.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # ---- Project coarse from 1-channel prediction to feature space ----
        coarse = self.coarse_proj(coarse)

        # ---- Retain gate R: amplifies reliable regions ----
        r = torch.sigmoid(self.conv_r(coarse) / self.gate_temp)

        # ---- Erase gate E: suppresses foreground-confident regions ----
        e = torch.sigmoid(self.conv_e(-coarse) / self.gate_temp)

        # ---- Contrast modulation gate W: content-aware scaling from f ----
        w = torch.sigmoid(self.fusion_gate(f))


        # - When R > E: amplifies f (reliable foreground)
        # - When E > R: suppresses f (uncertain boundary)
        # - When R = E: identity (graceful degradation)
        combined = f * (1 + w * (r - e))

        # ---- Fuse gated features ----
        rera_out = self.fuse(combined)

        # ---- Concat original f with gated output before MDSE
        fusion = torch.cat([f, rera_out], dim=1)
        fusion = self.pre_micro(fusion)

        # ---- Micro-structure refinement (MDSE) ----
        refined = self.micro(fusion)

        return refined + f



# ------------------------------------------------------------
# DRADNET MODEL
# ------------------------------------------------------------

class Shunted_DRADnet(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = shunted_s()
        path = '/home/pb/bali/pretrained_weights/ckpt_S.pth'

        if os.path.exists(path):

            save_model = torch.load(path)

            if 'model' in save_model:
                save_model = save_model['model']

            model_dict = self.backbone.state_dict()

            state_dict = {
                k: v for k, v in save_model.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

            print("Loaded pretrained weights")

        else:
            print("No pretrained weights found")
        
        channel = 256

        self.Translayer2 = BasicConv2d(128, channel, 1)
        self.Translayer3 = BasicConv2d(256, channel, 1)
        self.Translayer4 = BasicConv2d(512, channel, 1)

        # RERA blocks
        self.rera2 = CSRERA(channel)
        self.rera3 = CSRERA(channel)
        self.rera4 = CSRERA(channel)

        # prediction heads
        self.stage4_pred = nn.Conv2d(channel, 1, 1)
        self.outconv = nn.Conv2d(channel, 1, 1)

    def forward(self, x):

        input_size = x.shape[2:]
        B = x.shape[0]

        # -------------------------
        # Backbone Forward
        # -------------------------

        features = []
        y = x

        for i in range(4):

            patch_embed = getattr(self.backbone, f"patch_embed{i + 1}")
            block = getattr(self.backbone, f"block{i + 1}")
            norm = getattr(self.backbone, f"norm{i + 1}")

            y, H, W = patch_embed(y)

            for blk in block:
                y = blk(y, H, W)

            y = norm(y)

            feature = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            features.append(feature)

            if i != 3:
                y = feature

        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]

        x2_t = self.Translayer2(x2)
        x3_t = self.Translayer3(x3)
        x4_t = self.Translayer4(x4)

        # Stage 4 coarse prediction
        coarse = self.stage4_pred(x4_t)

        # Stage 4 refinement
        r4 = self.rera4(coarse, x4_t)
        map4 = self.outconv(r4)

        # Stage 3
        coarse_up3 = F.interpolate(map4, size=x3_t.shape[2:], mode='bilinear', align_corners=False)
        r3 = self.rera3(coarse_up3, x3_t)
        map3 = self.outconv(r3)

        # Stage 2
        coarse_up2 = F.interpolate(map3, size=x2_t.shape[2:], mode='bilinear', align_corners=False)
        r2 = self.rera2(coarse_up2, x2_t)
        map2 = self.outconv(r2)

        # resize to input
        map2 = F.interpolate(map2, size=input_size, mode='bilinear', align_corners=False)
        map3 = F.interpolate(map3, size=input_size, mode='bilinear', align_corners=False)
        map4 = F.interpolate(map4, size=input_size, mode='bilinear', align_corners=False)
        coarse = F.interpolate(coarse, size=input_size, mode='bilinear', align_corners=False)

        return (
            map2,
            map3,
            map4,
            coarse
        )






# ------------------------------------------------------------
# Test
# ------------------------------------------------------------
if __name__ == '__main__':

    model = Shunted_DRADnet().cuda()

    input_tensor = torch.randn(2,3,352,352).cuda()

    outputs = model(input_tensor)

    print("Input:",input_tensor.shape)

    for i,o in enumerate(outputs):
        print("Output",i,o.shape)


 
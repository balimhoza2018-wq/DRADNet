
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


class MDSE2(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # directional depthwise filters
        self.h   = nn.Conv2d(channels, channels, (1,3), padding=(0,1), groups=channels, bias=False)
        self.v   = nn.Conv2d(channels, channels, (3,1), padding=(1,0), groups=channels, bias=False)
        self.iso = nn.Conv2d(channels, channels, 3,     padding=1,     groups=channels, bias=False)

        # directional fusion → attention map A
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # centerness head → spatial confidence S
        self.centerness = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )

        # fusion projection: cat([x*S, x*A]) → channel
        # no ReLU — allows negative values so sigmoid gate can suppress
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # bounded residual gain
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):

        # directional attention A — what structure is present
        att = self.fuse(torch.cat([self.h(x), self.v(x), self.iso(x)], dim=1))

        # centerness S with LCN — where to trust structure
        center = self.centerness(x)
        center = center - F.avg_pool2d(center, kernel_size=7, stride=1, padding=3)
        center = torch.sigmoid(center)

        # bounded gain
        gain = torch.clamp(self.alpha, 0.0, 0.5)

        # decomposed branches
        x_center = x * center     # where branch
        x_att    = x * att        # what branch

        # learnable fusion of both cues — no ReLU allows suppression
        fusion = self.out_proj(torch.cat([x_center, x_att], dim=1))

        # multiplicative gate on x — can amplify or suppress spatially
        refined = gain * x * torch.sigmoid(fusion)

        # residual
        return x + refined


'''
class MicroStructureEnhancement2C2DV2(nn.Module):

    def __init__(self, channels, max_gain=0.5):
        super().__init__()

        self.max_gain = max_gain

        # Three-direction depthwise decomposition
        self.h = nn.Conv2d(channels, channels, (1, 3),
                           padding=(0, 1), groups=channels, bias=False)
        self.v = nn.Conv2d(channels, channels, (3, 1),
                           padding=(1, 0), groups=channels, bias=False)
        self.iso = nn.Conv2d(channels, channels, 3,
                             padding=1, groups=channels, bias=False)

        # Fuse with tanh → can suppress AND enhance
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Tanh()
        )

        # Per-channel centerness → richer spatial modulation
        self.centerness = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # Per-channel gain, starts near 0 (identity)
        self.alpha_logit = nn.Parameter(torch.full((channels, 1, 1), -4.0))

    def forward(self, x):
        h_feat = self.h(x)
        v_feat = self.v(x)
        i_feat = self.iso(x)

        # Centerness: local saliency map
        center = self.centerness(x)                              # [B, C, H, W]
        # Local contrast normalization — emphasize spatially distinctive regions
        center_lcn = center - F.avg_pool2d(center, 7, stride=1, padding=3)
        center_lcn = center_lcn / (center_lcn.abs().amax(dim=(2,3), keepdim=True) + 1e-6)
        center_lcn = torch.sigmoid(center_lcn)                   # [B, C, H, W]

        # Modulate directional features with centerness
        h_c = h_feat * (1.0 + center_lcn)
        v_c = v_feat * (1.0 + center_lcn)
        i_c = i_feat * (1.0 + center_lcn)

        # Fuse all directions → attention map in [-1, 1]
        att = self.fuse(torch.cat([h_c, v_c, i_c], dim=1))      # [B, C, H, W]

        # Per-channel bounded gain, starts near 0
        gain = self.max_gain * torch.sigmoid(self.alpha_logit)   # [C, 1, 1]

        # Multiplicative residual: amplify or suppress based on att sign
        refined = x + gain * x * att

        return refined
'''

# ------------------------------------------------------------
# Enhanced RERA
# ------------------------------------------------------------
class EnhancedRERA_MS_MS(nn.Module):

    def __init__(self, channel):
        super().__init__()

        self.gate_temp = 2.0

        self.coarse_proj = nn.Sequential(
            nn.Conv2d(1, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        self.conv_r = nn.Conv2d(channel, channel, 1)
        self.conv_e = nn.Conv2d(channel, channel, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        self.micro = MDSE(channel)

    def forward(self, coarse, f):

        if coarse.shape[2:] != f.shape[2:]:
            coarse = F.interpolate(
                coarse,
                size=f.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        coarse = self.coarse_proj(coarse)

        r = torch.sigmoid(self.conv_r(coarse) / self.gate_temp)
        e = torch.sigmoid(self.conv_e(-coarse) / self.gate_temp)

        r_f = r * f
        e_f = e * f

        combined = r_f-e_f

        rera_out = self.fuse(combined)

        refined = self.micro(rera_out)

        return refined + f
'''
r_f + e_f RESULTS
dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
-----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
CVC-300                0.884      0.816  0.863  0.926     0.957  0.007     0.849      0.956
CVC-ClinicDB           0.925      0.878  0.926  0.946     0.974  0.010     0.926      0.940
Kvasir                 0.922      0.876  0.920  0.929     0.960  0.021     0.948      0.913
CVC-ColonDB            0.795      0.719  0.784  0.861     0.897  0.033     0.827      0.825
ETIS-LaribPolypDB      0.758      0.683  0.724  0.853     0.874  0.019     0.728      0.847

e_f-r_f RESULTS
dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
-----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
CVC-300                0.879      0.813  0.859  0.924     0.950  0.008     0.852      0.950
CVC-ClinicDB           0.940      0.894  0.939  0.953     0.987  0.006     0.935      0.951
Kvasir                 0.904      0.854  0.900  0.914     0.953  0.026     0.941      0.894
CVC-ColonDB            0.767      0.688  0.755  0.843     0.879  0.037     0.809      0.788
ETIS-LaribPolypDB      0.769      0.691  0.734  0.856     0.885  0.020     0.746      0.861

r_f-e_f RESULTS
dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec
-----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
CVC-300                0.889      0.819  0.865  0.926     0.962  0.006     0.847      0.960
CVC-ClinicDB           0.924      0.874  0.918  0.945     0.974  0.007     0.911      0.956
Kvasir                 0.915      0.866  0.910  0.922     0.962  0.022     0.937      0.912
CVC-ColonDB            0.794      0.721  0.779  0.856     0.890  0.041     0.798      0.872
ETIS-LaribPolypDB      0.745      0.667  0.698  0.839     0.860  0.029     0.686      0.889

'''



class EnhancedRERA_MS_MS_ACARA(nn.Module):

    def __init__(self, channel):
        super().__init__()

        self.gate_temp = 2.0

        # ---- Coarse projection ----
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(1, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- Reverse & Edge gates ----
        self.conv_r = nn.Conv2d(channel, channel, 1)
        self.conv_e = nn.Conv2d(channel, channel, 1)

        # ---- Adaptive fusion weights (NEW) ----
        self.alpha = nn.Parameter(torch.tensor(1.0))  # contrast
        self.beta  = nn.Parameter(torch.tensor(0.5))  # smoothing

        # ---- RERA fusion conv ----
        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- Pre-MDSE fusion (concat) ----
        self.pre_micro = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- Micro-structure refinement ----
        self.micro = MDSE(channel)

    def forward(self, coarse, f):

        # ---- Resize coarse prediction ----
        if coarse.shape[2:] != f.shape[2:]:
            coarse = F.interpolate(
                coarse,
                size=f.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        coarse = self.coarse_proj(coarse)

        # ---- Reverse & Edge attention ----
        r = torch.sigmoid(self.conv_r(coarse) / self.gate_temp)
        e = torch.sigmoid(self.conv_e(-coarse) / self.gate_temp)

        r_f = r * f
        e_f = e * f

        # 🚀 ---- Adaptive fusion ----
        contrast = r_f - e_f      # (r - e)
        additive = r_f + e_f      # (r + e)

        combined = self.alpha * contrast + self.beta * additive

        # ---- Fuse ----
        rera_out = self.fuse(combined)

        # ---- Concat before MDSE ----
        fusion = torch.cat([f, rera_out], dim=1)
        fusion = self.pre_micro(fusion)

        # ---- Micro refinement ----
        refined = self.micro(fusion)

        return refined
        '''
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.883      0.819  0.867  0.929     0.951  0.009     0.859      0.950
        CVC-ClinicDB           0.940      0.894  0.940  0.952     0.986  0.006     0.937      0.952
        Kvasir                 0.919      0.871  0.914  0.925     0.963  0.020     0.941      0.916
        CVC-ColonDB            0.775      0.698  0.762  0.846     0.881  0.039     0.801      0.806
        ETIS-LaribPolypDB      0.723      0.660  0.695  0.836     0.869  0.020     0.708      0.806
        '''




class EnhancedRERA_MS_MS_ACARA2(nn.Module):

    def __init__(self, channel):
        super().__init__()

        self.gate_temp = 2.0

        # ---- Coarse projection ----
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(1, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- Reverse & Edge gates ----
        self.conv_r = nn.Conv2d(channel, channel, 1)
        self.conv_e = nn.Conv2d(channel, channel, 1)

        # 🚀 ---- Spatial adaptive fusion gate (NEW) ----
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

        # ---- RERA fusion conv ----
        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- Pre-MDSE fusion (concat) ----
        self.pre_micro = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- Micro-structure refinement ----
        self.micro = MDSE(channel)

    def forward(self, coarse, f):

        # ---- Resize coarse prediction ----
        if coarse.shape[2:] != f.shape[2:]:
            coarse = F.interpolate(
                coarse,
                size=f.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        coarse = self.coarse_proj(coarse)

        # ---- Reverse & Edge attention ----
        r = torch.sigmoid(self.conv_r(coarse) / self.gate_temp)
        e = torch.sigmoid(self.conv_e(-coarse) / self.gate_temp)

        r_f = r * f
        e_f = e * f

        # ---- Contrast & Additive components ----
        contrast = r_f - e_f      # discriminative
        additive = r_f + e_f      # smoothing

        # 🚀 ---- Spatial adaptive fusion ----
        w = self.fusion_gate(f)   # (B, C, H, W)

        combined = w * contrast + (1 - w) * additive

        # ---- Fuse ----
        rera_out = self.fuse(combined)

        # ---- Concat before MDSE ----
        fusion = torch.cat([f, rera_out], dim=1)
        fusion = self.pre_micro(fusion)

        # ---- Micro refinement ----
        refined = self.micro(fusion)

        return refined
        '''
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.879      0.805  0.850  0.923     0.954  0.008     0.826      0.971
        CVC-ClinicDB           0.931      0.878  0.925  0.943     0.984  0.008     0.913      0.958
        Kvasir                 0.924      0.874  0.910  0.927     0.964  0.022     0.919      0.946
        CVC-ColonDB            0.792      0.712  0.770  0.854     0.898  0.032     0.793      0.849
        ETIS-LaribPolypDB      0.716      0.639  0.670  0.827     0.837  0.029     0.664      0.870
        '''

class EnhancedRERA_MS_MS_New(nn.Module):
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

        # ---- Pre-MDSE fusion (concat f + rera_out) ----
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

        # ---- Differential dual-gated modulation ----
        # combined = f * (1 + W * (R - E))
        # - When R > E: amplifies f (reliable foreground)
        # - When E > R: suppresses f (uncertain boundary)
        # - When R = E: identity (graceful degradation)
        combined = f * (1 + w * (r - e))

        # ---- Fuse gated features ----
        rera_out = self.fuse(combined)

        # ---- Concat original f with gated output before MDSE ----
        # preserves raw features alongside refined features
        fusion = torch.cat([f, rera_out], dim=1)
        fusion = self.pre_micro(fusion)

        # ---- Micro-structure refinement (MDSE) ----
        refined = self.micro(fusion)

        # ---- Residual connection: stabilizes training ----
        # if MDSE output is poor early in training, f is preserved
        return refined + f
        '''
        35 epochs
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.883      0.818  0.864  0.929     0.951  0.009     0.856      0.952
        CVC-ClinicDB           0.940      0.893  0.942  0.951     0.986  0.006     0.943      0.945
        Kvasir                 0.918      0.866  0.912  0.924     0.954  0.026     0.945      0.915
        CVC-ColonDB            0.784      0.705  0.771  0.855     0.894  0.031     0.815      0.811
        ETIS-LaribPolypDB      0.779      0.705  0.744  0.866     0.894  0.020     0.742      0.875
        '''


class EnhancedRERA_MS_MS_New2(nn.Module):
    def __init__(self, channel):
        super().__init__()

        # gate temperature — registered buffer so it saves with state_dict
        self.register_buffer('gate_temp', torch.tensor(2.0))

        # coarse projection: 1-channel prediction → feature space
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(1, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # retain and erase gates — separate projections, bias=True (explicit)
        self.conv_r = nn.Conv2d(channel, channel, 1, bias=True)
        self.conv_e = nn.Conv2d(channel, channel, 1, bias=True)

        # contrast modulation gate from f
        self.fusion_gate = nn.Conv2d(channel, channel, 1, bias=False)

        # RERA fusion conv
        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # pre-MDSE fusion: concat [f, rera_out] → channel
        self.pre_micro = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # micro-structure refinement
        self.micro = MDSE(channel)

    def forward(self, coarse, f):

        # resize coarse to match f spatial dimensions
        if coarse.shape[2:] != f.shape[2:]:
            coarse = F.interpolate(
                coarse, size=f.shape[2:],
                mode='bilinear', align_corners=False
            )

        # project coarse prediction into feature space
        coarse = self.coarse_proj(coarse)

        # retain gate R — amplifies reliable regions
        r = torch.sigmoid(self.conv_r(coarse) / self.gate_temp)

        # erase gate E — suppresses foreground-confident regions
        # conv_e and conv_r have separate weights → R + E ≠ 1
        e = torch.sigmoid(self.conv_e(-coarse) / self.gate_temp)

        # contrast modulation gate W — content-aware scaling from f
        w = torch.sigmoid(self.fusion_gate(f) / self.gate_temp)

        # differential dual-gated modulation
        # modulation bounded to [0, 2] for stability
        modulation = torch.clamp(1 + w * (r - e), min=0.0, max=2.0)
        combined = f * modulation

        # fuse gated features
        rera_out = self.fuse(combined)

        # concat original f with gated output → preserves raw features
        fusion = self.pre_micro(torch.cat([f, rera_out], dim=1))

        # micro-structure refinement
        refined = self.micro(fusion)

        # residual: stabilizes training, guarantees gradient flow
        return refined + f
        '''
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.880      0.811  0.857  0.923     0.957  0.008     0.841      0.956
        CVC-ClinicDB           0.934      0.887  0.932  0.950     0.979  0.006     0.926      0.957
        Kvasir                 0.922      0.875  0.917  0.929     0.961  0.020     0.944      0.922
        CVC-ColonDB            0.788      0.710  0.773  0.852     0.897  0.034     0.809      0.815
        ETIS-LaribPolypDB      0.769      0.691  0.728  0.859     0.885  0.017     0.725      0.874
        '''


class EnhancedRERA_MS_MS_New3(nn.Module):
    def __init__(self, channel):
        super().__init__()

        # gate temperature — registered buffer
        self.register_buffer('gate_temp', torch.tensor(2.0))

        # coarse projection: 1-channel prediction → feature space
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(1, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # retain and erase gates — separate projections
        self.conv_r = nn.Conv2d(channel, channel, 1, bias=True)
        self.conv_e = nn.Conv2d(channel, channel, 1, bias=True)

        # contrast modulation gate from f — no temperature scaling
        self.fusion_gate = nn.Conv2d(channel, channel, 1, bias=False)

        # multi-scale dilated depthwise enrichment of f
        self.dw_r3 = nn.Conv2d(channel, channel, 3,
                                padding=3, dilation=3,
                                groups=channel, bias=False)
        self.dw_r5 = nn.Conv2d(channel, channel, 3,
                                padding=5, dilation=5,
                                groups=channel, bias=False)
        self.dw_r7 = nn.Conv2d(channel, channel, 3,
                                padding=7, dilation=7,
                                groups=channel, bias=False)
        self.dw_fuse = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # RERA fusion conv
        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # pre-MDSE fusion: concat [f, rera_out] → channel
        self.pre_micro = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # micro-structure refinement
        self.micro = MDSE(channel)

    def forward(self, coarse, f):

        # resize coarse to match f spatial dimensions
        if coarse.shape[2:] != f.shape[2:]:
            coarse = F.interpolate(
                coarse, size=f.shape[2:],
                mode='bilinear', align_corners=False
            )

        # project coarse prediction into feature space
        coarse = self.coarse_proj(coarse)

        # retain gate R — amplifies reliable regions
        r = torch.sigmoid(self.conv_r(coarse) / self.gate_temp)

        # erase gate E — suppresses foreground-confident regions
        # conv_e and conv_r have separate weights → R + E ≠ 1
        e = torch.sigmoid(self.conv_e(-coarse) / self.gate_temp)

        # contrast modulation gate W — content-aware scaling from f
        # no temperature scaling: f activations are already stable
        w = torch.sigmoid(self.fusion_gate(f))

        # enrich f with multi-scale context before gating
        # rates 3, 5, 7 capture local, medium, and wide structural context
        f_rich = self.dw_fuse(torch.cat([
            self.dw_r3(f),
            self.dw_r5(f),
            self.dw_r7(f)
        ], dim=1))

        # differential dual-gated modulation on enriched f
        # bounded to [0, 2] for training stability
        # R > E: amplifies f_rich (reliable foreground)
        # E > R: suppresses f_rich (uncertain boundary)
        # R = E: identity, graceful degradation
        modulation = torch.clamp(1 + w * (r - e), min=0.0, max=2.0)
        combined = f_rich * modulation

        # fuse gated features
        rera_out = self.fuse(combined)

        # concat original f (not f_rich) with gated output
        # original f is preserved here — f_rich only used for gating
        fusion = self.pre_micro(torch.cat([f, rera_out], dim=1))

        # micro-structure refinement
        refined = self.micro(fusion)

        # residual with original f (not f_rich)
        # f_rich enrichment is internal — clean skip is preserved
        return refined + f
        '''
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.895      0.828  0.875  0.932     0.965  0.006     0.857      0.962
        CVC-ClinicDB           0.933      0.883  0.933  0.945     0.980  0.010     0.933      0.945
        Kvasir                 0.916      0.864  0.911  0.923     0.958  0.025     0.942      0.912
        CVC-ColonDB            0.777      0.695  0.759  0.842     0.880  0.039     0.799      0.836
        ETIS-LaribPolypDB      0.785      0.702  0.741  0.866     0.904  0.014     0.735      0.891

    '''



class EnhancedRERA_MS_MS_New34(nn.Module):
    def __init__(self, channel):
        super().__init__()

        # gate temperature — registered buffer
        self.register_buffer('gate_temp', torch.tensor(2.0))

        # coarse projection: 1-channel prediction → feature space
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(1, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # retain and erase gates — separate projections
        self.conv_r = nn.Conv2d(channel, channel, 1, bias=True)
        self.conv_e = nn.Conv2d(channel, channel, 1, bias=True)

        # contrast modulation gate from f — no temperature scaling
        self.fusion_gate = nn.Conv2d(channel, channel, 1, bias=False)

        # multi-scale dilated depthwise enrichment of f
        self.dw_r1 = nn.Conv2d(channel, channel, 3,
                                padding=1, dilation=1,
                                groups=channel, bias=False)
        self.dw_r3 = nn.Conv2d(channel, channel, 3,
                                padding=3, dilation=3,
                                groups=channel, bias=False)
        self.dw_fuse = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # RERA fusion conv
        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # pre-MDSE fusion: concat [f, rera_out] → channel
        self.pre_micro = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # micro-structure refinement
        self.micro = MDSE(channel)

    def forward(self, coarse, f):

        # resize coarse to match f spatial dimensions
        if coarse.shape[2:] != f.shape[2:]:
            coarse = F.interpolate(
                coarse, size=f.shape[2:],
                mode='bilinear', align_corners=False
            )

        # project coarse prediction into feature space
        coarse = self.coarse_proj(coarse)

        # retain gate R — amplifies reliable regions
        r = torch.sigmoid(self.conv_r(coarse) / self.gate_temp)

        # erase gate E — suppresses foreground-confident regions
        # conv_e and conv_r have separate weights → R + E ≠ 1
        e = torch.sigmoid(self.conv_e(-coarse) / self.gate_temp)

        # contrast modulation gate W — content-aware scaling from f
        # no temperature scaling: f activations are already stable
        w = torch.sigmoid(self.fusion_gate(f))

        # enrich f with multi-scale context before gating
        # rates 3, 5, 7 capture local, medium, and wide structural context
        f_rich = self.dw_fuse(torch.cat([
            self.dw_r1(f),
            self.dw_r3(f)
        ], dim=1))

        # differential dual-gated modulation on enriched f
        # bounded to [0, 2] for training stability
        # R > E: amplifies f_rich (reliable foreground)
        # E > R: suppresses f_rich (uncertain boundary)
        # R = E: identity, graceful degradation
        modulation = torch.clamp(1 + w * (r - e), min=0.0, max=2.0)
        combined = f_rich * modulation

        # fuse gated features
        rera_out = self.fuse(combined)

        # concat original f (not f_rich) with gated output
        # original f is preserved here — f_rich only used for gating
        fusion = self.pre_micro(torch.cat([f_rich, rera_out], dim=1))

        # micro-structure refinement
        refined = self.micro(fusion)

        # residual with original f (not f_rich)
        # f_rich enrichment is internal — clean skip is preserved
        return refined + f
        '''
        nofusionn of rera and f, only rera
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.892      0.820  0.868  0.928     0.967  0.007     0.849      0.961
        CVC-ClinicDB           0.928      0.880  0.928  0.944     0.976  0.008     0.925      0.949
        Kvasir                 0.912      0.861  0.907  0.922     0.956  0.023     0.937      0.908
        CVC-ColonDB            0.760      0.683  0.746  0.838     0.881  0.036     0.792      0.780
        ETIS-LaribPolypDB      0.714      0.636  0.673  0.823     0.850  0.024     0.673      0.831
        '''


class EnhancedRERA_MS_MS_New4(nn.Module):
    """
    Dual-Stage Multi-Scale Contrast-Modulated Reverse Attention (RERA)

    Key components:
    1. Reverse Attention (R, E):
        - R: retain (foreground-enhancing)
        - E: erase (background-suppressing)

    2. Contrast Modulation:
        - Applies spatially adaptive contrast: (r - e)
        - Controlled by gate w
        - Stable scaling: f * (1 + 0.5 * w * (r - e))

    3. Multi-Scale Feature Enrichment:
        - Stage 1 (pre-gating): structure-aware (dilations 3,5,7)
        - Stage 2 (pre-fusion): detail-aware (dilations 1,3,5)

    4. MDSE Refinement:
        - Enhances micro-structures and boundaries

    5. Clean Residual:
        - Final output preserves original feature information
    """

    def __init__(self, channel):
        super().__init__()

        # -------------------------------------------------
        # Gate temperature (non-trainable, stable scaling)
        # -------------------------------------------------
        self.register_buffer('gate_temp', torch.tensor(2.0))

        # -------------------------------------------------
        # Coarse prediction projection (1 → C)
        # -------------------------------------------------
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(1, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------
        # Reverse Attention Gates
        # -------------------------------------------------
        self.conv_r = nn.Conv2d(channel, channel, 1, bias=False)
        self.conv_e = nn.Conv2d(channel, channel, 1, bias=False)

        # -------------------------------------------------
        # Contrast modulation gate (spatial + channel aware)
        # -------------------------------------------------
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel)
        )

        # =================================================
        # Stage 1: Multi-scale structure context (pre-gating)
        # =================================================
        self.dw_s3 = nn.Conv2d(channel, channel, 3,
                              padding=3, dilation=3,
                              groups=channel, bias=False)
        self.dw_s5 = nn.Conv2d(channel, channel, 3,
                              padding=5, dilation=5,
                              groups=channel, bias=False)
        self.dw_s7 = nn.Conv2d(channel, channel, 3,
                              padding=7, dilation=7,
                              groups=channel, bias=False)

        self.struct_fuse = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # =================================================
        # Stage 2: Multi-scale detail context (pre-fusion)
        # =================================================
        self.dw_d1 = nn.Conv2d(channel, channel, 3,
                              padding=1, dilation=1,
                              groups=channel, bias=False)
        self.dw_d3 = nn.Conv2d(channel, channel, 3,
                              padding=3, dilation=3,
                              groups=channel, bias=False)
        self.dw_d5 = nn.Conv2d(channel, channel, 3,
                              padding=5, dilation=5,
                              groups=channel, bias=False)

        self.detail_fuse = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------
        # RERA feature fusion
        # -------------------------------------------------
        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------
        # Pre-MDSE fusion (concat + projection)
        # -------------------------------------------------
        self.pre_micro = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------
        # Micro-structure enhancement
        # -------------------------------------------------
        self.micro = MDSE(channel)

    def forward(self, coarse, f):
        """
        Args:
            coarse: (B, 1, H, W) coarse prediction
            f:      (B, C, H, W) feature map

        Returns:
            refined feature map (B, C, H, W)
        """

        # -------------------------------------------------
        # Resize coarse prediction
        # -------------------------------------------------
        if coarse.shape[2:] != f.shape[2:]:
            coarse = F.interpolate(
                coarse, size=f.shape[2:],
                mode='bilinear', align_corners=False
            )

        # -------------------------------------------------
        # Project coarse prediction to feature space
        # -------------------------------------------------
        coarse = self.coarse_proj(coarse)

        # -------------------------------------------------
        # Reverse Attention Gates
        # -------------------------------------------------
        r = torch.sigmoid(self.conv_r(coarse) / self.gate_temp)
        e = torch.sigmoid(self.conv_e(-coarse) / self.gate_temp)

        # -------------------------------------------------
        # Contrast modulation gate
        # -------------------------------------------------
        w = torch.sigmoid(self.fusion_gate(f))

        # =================================================
        # Stage 1: Structure-aware multi-scale features
        # =================================================
        f_struct = self.struct_fuse(torch.cat([
            self.dw_s3(f),
            self.dw_s5(f),
            self.dw_s7(f)
        ], dim=1))

        # =================================================
        # Stage 2: Detail-aware multi-scale features
        # =================================================
        f_detail = self.detail_fuse(torch.cat([
            self.dw_d1(f),
            self.dw_d3(f),
            self.dw_d5(f)
        ], dim=1))
        # -------------------------------------------------
        # Contrast modulation (stable form)
        # -------------------------------------------------
        modulation = 1 + 0.5 * w * (r - e)
        combined = f_detail * modulation

        # -------------------------------------------------
        # Fuse RERA features
        # -------------------------------------------------
        rera_out = self.fuse(combined)


        # -------------------------------------------------
        # Concatenate detail + RERA output
        # -------------------------------------------------
        fusion = self.pre_micro(torch.cat([f_struct, rera_out], dim=1))

        # -------------------------------------------------
        # Micro-structure refinement
        # -------------------------------------------------
        refined = self.micro(fusion)

        # -------------------------------------------------
        # Residual connection
        # -------------------------------------------------
        return refined + f
        '''
        f_detail at rera_out and f_struct at modulation
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.897      0.828  0.874  0.931     0.968  0.006     0.855      0.965
        CVC-ClinicDB           0.936      0.890  0.934  0.949     0.980  0.007     0.927      0.957
        Kvasir                 0.906      0.857  0.902  0.918     0.954  0.026     0.935      0.897
        CVC-ColonDB            0.811      0.733  0.795  0.864     0.917  0.026     0.824      0.839
        ETIS-LaribPolypDB      0.769      0.689  0.729  0.858     0.891  0.019     0.734      0.872

        f_struct at rera_out and f_detail at modulation
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.894      0.825  0.873  0.931     0.965  0.007     0.859      0.956
        CVC-ClinicDB           0.925      0.876  0.922  0.947     0.975  0.008     0.920      0.950
        Kvasir                 0.905      0.852  0.899  0.916     0.950  0.028     0.940      0.898
        CVC-ColonDB            0.804      0.726  0.790  0.866     0.907  0.030     0.827      0.829
        ETIS-LaribPolypDB      0.744      0.673  0.708  0.844     0.874  0.019     0.708      0.860
        '''
                     
class EnhancedRERA_MS_MS_New5(nn.Module):
    def __init__(self, channel):
        super().__init__()

        # gate temperature — registered buffer
        self.register_buffer('gate_temp', torch.tensor(2.0))

        # coarse projection: 1-channel prediction → feature space
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(1, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # retain and erase gates — separate projections
        self.conv_r = nn.Conv2d(channel, channel, 1, bias=True)
        self.conv_e = nn.Conv2d(channel, channel, 1, bias=True)

        # contrast modulation gate from f — no temperature scaling
        self.fusion_gate = nn.Conv2d(channel, channel, 1, bias=False)

        # multi-scale dilated depthwise enrichment of f
        self.dw_r1 = nn.Conv2d(channel, channel, 3,
                                padding=1, dilation=1,
                                groups=channel, bias=False)
        self.dw_r3 = nn.Conv2d(channel, channel, 3,
                                padding=3, dilation=3,
                                groups=channel, bias=False)
        self.dw_r5 = nn.Conv2d(channel, channel, 3,
                                padding=5, dilation=5,
                                groups=channel, bias=False)
        self.dw_fuse = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # RERA fusion conv
        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # pre-MDSE fusion: concat [f, rera_out] → channel
        self.pre_micro = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # micro-structure refinement
        self.micro = MDSE2(channel)

    def forward(self, coarse, f):

        # resize coarse to match f spatial dimensions
        if coarse.shape[2:] != f.shape[2:]:
            coarse = F.interpolate(
                coarse, size=f.shape[2:],
                mode='bilinear', align_corners=False
            )

        # project coarse prediction into feature space
        coarse = self.coarse_proj(coarse)

        # retain gate R — amplifies reliable regions
        r = torch.sigmoid(self.conv_r(coarse) / self.gate_temp)

        # erase gate E — suppresses foreground-confident regions
        # conv_e and conv_r have separate weights → R + E ≠ 1
        e = torch.sigmoid(self.conv_e(-coarse) / self.gate_temp)

        # contrast modulation gate W — content-aware scaling from f
        # no temperature scaling: f activations are already stable
        w = torch.sigmoid(self.fusion_gate(f))

        # enrich f with multi-scale context before gating
        # rates 3, 5, 7 capture local, medium, and wide structural context
        f_rich = self.dw_fuse(torch.cat([
            self.dw_r1(f),
            self.dw_r3(f),
            self.dw_r5(f)
        ], dim=1))

        # differential dual-gated modulation on enriched f
        # bounded to [0, 2] for training stability
        # R > E: amplifies f_rich (reliable foreground)
        # E > R: suppresses f_rich (uncertain boundary)
        # R = E: identity, graceful degradation
        modulation = torch.clamp(1 + w * (r - e), min=0.0, max=2.0)
        combined = f * modulation

        # fuse gated features
        rera_out = self.fuse(combined)

        # concat original f (not f_rich) with gated output
        # original f is preserved here — f_rich only used for gating
        fusion = self.pre_micro(torch.cat([f_rich, rera_out], dim=1))

        # micro-structure refinement
        refined = self.micro(fusion)

        # residual with original f (not f_rich)
        # f_rich enrichment is internal — clean skip is preserved
        return refined + f
        '''
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.893      0.820  0.871  0.928     0.970  0.006     0.859      0.950
        CVC-ClinicDB           0.924      0.876  0.927  0.940     0.975  0.014     0.943      0.927
        Kvasir                 0.895      0.839  0.879  0.906     0.947  0.033     0.902      0.912
        CVC-ColonDB            0.776      0.697  0.763  0.847     0.889  0.036     0.808      0.803
        ETIS-LaribPolypDB      0.752      0.680  0.721  0.849     0.878  0.021     0.725      0.835
        '''

class EnhancedRERA_MS_MS_New6(nn.Module):
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

        # ---- Pre-MDSE fusion (concat f + rera_out) ----
        self.pre_micro = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- Micro-structure refinement ----
        self.micro = MDSE2(channel)

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

        # ---- Differential dual-gated modulation ----
        # combined = f * (1 + W * (R - E))
        # - When R > E: amplifies f (reliable foreground)
        # - When E > R: suppresses f (uncertain boundary)
        # - When R = E: identity (graceful degradation)
        combined = f * (1 + w * (r - e))

        # ---- Fuse gated features ----
        rera_out = self.fuse(combined)

        # ---- Concat original f with gated output before MDSE ----
        # preserves raw features alongside refined features
        fusion = torch.cat([f, rera_out], dim=1)
        fusion = self.pre_micro(fusion)

        # ---- Micro-structure refinement (MDSE) ----
        refined = self.micro(fusion)

        # ---- Residual connection: stabilizes training ----
        # if MDSE output is poor early in training, f is preserved
        return refined + f
        '''
        with modified loss
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.882      0.821  0.868  0.929     0.951  0.008     0.859      0.949
        CVC-ClinicDB           0.934      0.884  0.934  0.947     0.983  0.009     0.938      0.940
        Kvasir                 0.916      0.867  0.915  0.927     0.959  0.021     0.955      0.900
        CVC-ColonDB            0.798      0.718  0.785  0.858     0.891  0.040     0.830      0.842
        ETIS-LaribPolypDB      0.739      0.665  0.702  0.838     0.862  0.033     0.707      0.854

        original loss
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.873      0.791  0.837  0.911     0.954  0.008     0.815      0.969
        CVC-ClinicDB           0.936      0.887  0.932  0.950     0.986  0.007     0.917      0.964
        Kvasir                 0.916      0.865  0.905  0.922     0.962  0.025     0.922      0.925
        CVC-ColonDB            0.785      0.701  0.761  0.851     0.894  0.029     0.796      0.837
        ETIS-LaribPolypDB      0.747      0.662  0.694  0.840     0.855  0.027     0.681      0.905
        '''

class EnhancedRERA_MS_MS_New8(nn.Module):
    """
    Dual-Stage Multi-Scale Contrast-Modulated Reverse Attention (RERA)

    Key components:
    1. Reverse Attention (R, E):
        - R: retain (foreground-enhancing)
        - E: erase (background-suppressing)

    2. Contrast Modulation:
        - Applies spatially adaptive contrast: (r - e)
        - Controlled by gate w
        - Stable scaling: f * (1 + 0.5 * w * (r - e))

    3. Multi-Scale Feature Enrichment:
        - Stage 1 (pre-gating): structure-aware (dilations 3,5,7)
        - Stage 2 (pre-fusion): detail-aware (dilations 1,3,5)

    4. MDSE Refinement:
        - Enhances micro-structures and boundaries

    5. Clean Residual:
        - Final output preserves original feature information
    """

    def __init__(self, channel):
        super().__init__()

        # -------------------------------------------------
        # Gate temperature (non-trainable, stable scaling)
        # -------------------------------------------------
        self.register_buffer('gate_temp', torch.tensor(2.0))

        # -------------------------------------------------
        # Coarse prediction projection (1 → C)
        # -------------------------------------------------
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(1, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------
        # Reverse Attention Gates
        # -------------------------------------------------
        self.conv_r = nn.Conv2d(channel, channel, 1, bias=False)
        self.conv_e = nn.Conv2d(channel, channel, 1, bias=False)

        # -------------------------------------------------
        # Contrast modulation gate (spatial + channel aware)
        # -------------------------------------------------
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel)
        )

        # =================================================
        # Stage 1: Multi-scale structure context (pre-gating)
        # =================================================
        self.dw_s3 = nn.Conv2d(channel, channel, 3,
                              padding=3, dilation=3,
                              groups=channel, bias=False)
        self.dw_s5 = nn.Conv2d(channel, channel, 3,
                              padding=5, dilation=5,
                              groups=channel, bias=False)
        self.dw_s7 = nn.Conv2d(channel, channel, 3,
                              padding=7, dilation=7,
                              groups=channel, bias=False)

        self.struct_fuse = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # =================================================
        # Stage 2: Multi-scale detail context (pre-fusion)
        # =================================================
        self.dw_d1 = nn.Conv2d(channel, channel, 3,
                              padding=1, dilation=1,
                              groups=channel, bias=False)
        self.dw_d3 = nn.Conv2d(channel, channel, 3,
                              padding=3, dilation=3,
                              groups=channel, bias=False)
        self.dw_d5 = nn.Conv2d(channel, channel, 3,
                              padding=5, dilation=5,
                              groups=channel, bias=False)

        self.detail_fuse = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------
        # RERA feature fusion
        # -------------------------------------------------
        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------
        # Pre-MDSE fusion (concat + projection)
        # -------------------------------------------------
        self.pre_micro = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------
        # Micro-structure enhancement
        # -------------------------------------------------
        self.micro = MDSE(channel)

    def forward(self, coarse, f):
        """
        Args:
            coarse: (B, 1, H, W) coarse prediction
            f:      (B, C, H, W) feature map

        Returns:
            refined feature map (B, C, H, W)
        """

        # -------------------------------------------------
        # Resize coarse prediction
        # -------------------------------------------------
        if coarse.shape[2:] != f.shape[2:]:
            coarse = F.interpolate(
                coarse, size=f.shape[2:],
                mode='bilinear', align_corners=False
            )

        # -------------------------------------------------
        # Project coarse prediction to feature space
        # -------------------------------------------------
        coarse = self.coarse_proj(coarse)

        # -------------------------------------------------
        # Reverse Attention Gates
        # -------------------------------------------------
        r = torch.sigmoid(self.conv_r(coarse) / self.gate_temp)
        e = torch.sigmoid(self.conv_e(-coarse) / self.gate_temp)

        # -------------------------------------------------
        # Contrast modulation gate
        # -------------------------------------------------
        w = torch.sigmoid(self.fusion_gate(f))

        # =================================================
        # Stage 1: Structure-aware multi-scale features
        # =================================================
        f_struct = self.struct_fuse(torch.cat([
            self.dw_s3(f),
            self.dw_s5(f),
            self.dw_s7(f)
        ], dim=1))

        # =================================================
        # Stage 2: Detail-aware multi-scale features
        # =================================================
        f_detail = self.detail_fuse(torch.cat([
            self.dw_d1(f),
            self.dw_d3(f),
            self.dw_d5(f)
        ], dim=1))
        # -------------------------------------------------
        # Contrast modulation (stable form)
        # -------------------------------------------------
        modulation = 1 + w * (r - e)
        combined = f_struct * modulation

        # -------------------------------------------------
        # Fuse RERA features
        # -------------------------------------------------
        rera_out = self.fuse(combined)


        # -------------------------------------------------
        # Concatenate detail + RERA output
        # -------------------------------------------------
        fusion = self.pre_micro(torch.cat([f_detail, rera_out], dim=1))

        # -------------------------------------------------
        # Micro-structure refinement
        # -------------------------------------------------
        refined = self.micro(fusion)

        # -------------------------------------------------
        # Residual connection
        # -------------------------------------------------
        return refined + f




class EnhancedRERA_MS_MS_SoftmaxRE(nn.Module):
    """
    Dual-Stage Multi-Scale Contrast-Modulated Reverse Attention (Softmax R/E)

    Key change:
    - Replace independent sigmoid gates with softmax competition:
        r + e = 1

    This enforces stronger foreground-background discrimination.
    """

    def __init__(self, channel):
        super().__init__()

        # ---- Gate temperature ----
        self.register_buffer('gate_temp', torch.tensor(2.0))

        # ---- Coarse projection ----
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(1, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- R/E projections ----
        self.conv_r = nn.Conv2d(channel, channel, 1, bias=False)
        self.conv_e = nn.Conv2d(channel, channel, 1, bias=False)

        # ---- Contrast modulation gate (unchanged) ----
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel)
        )

        # =================================================
        # Stage 1: Structure-aware multi-scale features
        # =================================================
        self.dw_s3 = nn.Conv2d(channel, channel, 3,
                              padding=3, dilation=3,
                              groups=channel, bias=False)
        self.dw_s5 = nn.Conv2d(channel, channel, 3,
                              padding=5, dilation=5,
                              groups=channel, bias=False)
        self.dw_s7 = nn.Conv2d(channel, channel, 3,
                              padding=7, dilation=7,
                              groups=channel, bias=False)

        self.struct_fuse = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # =================================================
        # Stage 2: Detail-aware multi-scale features
        # =================================================
        self.dw_d1 = nn.Conv2d(channel, channel, 3,
                              padding=1, dilation=1,
                              groups=channel, bias=False)
        self.dw_d3 = nn.Conv2d(channel, channel, 3,
                              padding=3, dilation=3,
                              groups=channel, bias=False)
        self.dw_d5 = nn.Conv2d(channel, channel, 3,
                              padding=5, dilation=5,
                              groups=channel, bias=False)

        self.detail_fuse = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- RERA fusion ----
        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- Pre-MDSE fusion ----
        self.pre_micro = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # ---- Micro refinement ----
        self.micro = MDSE(channel)

    def forward(self, coarse, f):

        # ---- Resize coarse ----
        if coarse.shape[2:] != f.shape[2:]:
            coarse = F.interpolate(
                coarse, size=f.shape[2:],
                mode='bilinear', align_corners=False
            )

        coarse = self.coarse_proj(coarse)

        # =================================================
        # 🔥 Softmax-based R/E competition
        # =================================================
        r_logits = self.conv_r(coarse) / self.gate_temp
        e_logits = self.conv_e(-coarse) / self.gate_temp

        # Stack → (B, 2C, H, W)
        re_logits = torch.cat([r_logits, e_logits], dim=1)

        # Softmax across R/E pairs
        re = torch.softmax(re_logits, dim=1)

        # Split back
        r, e = torch.chunk(re, 2, dim=1)

        # ---- Contrast modulation gate ----
        w = torch.sigmoid(self.fusion_gate(f))

        # =================================================
        # Stage 1: Structure features
        # =================================================
        f_struct = self.struct_fuse(torch.cat([
            self.dw_s3(f),
            self.dw_s5(f),
            self.dw_s7(f)
        ], dim=1))

        # =================================================
        # Stage 2: Detail features (used for modulation)
        # =================================================
        f_detail = self.detail_fuse(torch.cat([
            self.dw_d1(f),
            self.dw_d3(f),
            self.dw_d5(f)
        ], dim=1))

        # ---- Contrast modulation (unchanged) ----
        modulation = 1 + w * (r - e)
        combined = f_detail * modulation

        # ---- Fuse ----
        rera_out = self.fuse(combined)

        # ---- Final fusion ----
        fusion = self.pre_micro(torch.cat([f_struct, rera_out], dim=1))

        # ---- Micro refinement ----
        refined = self.micro(fusion)

        return refined + f
        '''
        with 0.5
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.899      0.827  0.875  0.930     0.971  0.006     0.846      0.975
        CVC-ClinicDB           0.936      0.886  0.931  0.947     0.984  0.007     0.913      0.968
        Kvasir                 0.914      0.864  0.906  0.923     0.964  0.020     0.926      0.919
        CVC-ColonDB            0.792      0.714  0.775  0.858     0.899  0.031     0.802      0.841
        ETIS-LaribPolypDB      0.769      0.688  0.724  0.858     0.882  0.023     0.713      0.910

        without 0.5
        dataset              meanDic    meanIoU    wFm     Sm    meanEm    mae    meanPr    meanRec                            
        -----------------  ---------  ---------  -----  -----  --------  -----  --------  ---------
        CVC-300                0.897      0.827  0.876  0.931     0.974  0.006     0.856      0.959
        CVC-ClinicDB           0.930      0.883  0.928  0.947     0.977  0.012     0.931      0.946
        Kvasir                 0.923      0.874  0.917  0.930     0.964  0.019     0.940      0.924
        CVC-ColonDB            0.807      0.724  0.789  0.859     0.910  0.034     0.819      0.851
        ETIS-LaribPolypDB      0.721      0.641  0.674  0.826     0.854  0.028     0.666      0.865
        '''


# ------------------------------------------------------------
# PVT PraNet V2 (Stage-4 Decoder)
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
        '''
        self.backbone = pvt_v2_b2()

        path = '/home/pb/bali/pretrained_weights/pvt_v2_b2.pth'

        if os.path.exists(path):

            save_model = torch.load(path)

            model_dict = self.backbone.state_dict()

            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}

            model_dict.update(state_dict)

            self.backbone.load_state_dict(model_dict)

            print(f"Loaded pretrained weights from {path}")
        '''
        channel = 256

        self.Translayer2 = BasicConv2d(128, channel, 1)
        self.Translayer3 = BasicConv2d(256, channel, 1)#320 for pvt
        self.Translayer4 = BasicConv2d(512, channel, 1)

        # RERA blocks
        self.rera2 = EnhancedRERA_MS_MS_New34(channel)
        self.rera3 = EnhancedRERA_MS_MS_New34(channel)
        self.rera4 = EnhancedRERA_MS_MS_New34(channel)

        # prediction heads
        self.stage4_pred = nn.Conv2d(channel,1,1)
        self.outconv = nn.Conv2d(channel,1,1)

    def forward(self, x):

        #input_size = x.shape[2:]

        #pvt = self.backbone(x)
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
            coarse,
            1 - torch.sigmoid(map2),
            1 - torch.sigmoid(map3),
            1 - torch.sigmoid(map4),
            1 - torch.sigmoid(coarse)
        )




#BASIC DECODER
class Shunted_DRADnet_NORMAL(nn.Module):

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

        # Translayers
        self.Translayer2 = BasicConv2d(128, channel, 1)
        self.Translayer3 = BasicConv2d(256, channel, 1)
        self.Translayer4 = BasicConv2d(512, channel, 1)

        # Simple decoder
        self.decoder4 = BasicConv2d(channel, channel, 3, padding=1)
        self.decoder3 = BasicConv2d(channel, channel, 3, padding=1)
        self.decoder2 = BasicConv2d(channel, channel, 3, padding=1)

        # Prediction heads
        self.stage4_pred = nn.Conv2d(channel, 1, 1)
        self.pred4 = nn.Conv2d(channel, 1, 1)
        self.pred3 = nn.Conv2d(channel, 1, 1)
        self.pred2 = nn.Conv2d(channel, 1, 1)

    def forward(self, x):

        input_size = x.shape[2:]
        B = x.shape[0]

        # Manual backbone forward (same as working version)
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

        x2 = self.Translayer2(features[1])
        x3 = self.Translayer3(features[2])
        x4 = self.Translayer4(features[3])

        # Coarse prediction
        coarse = self.stage4_pred(x4)

        # Stage 4
        d4 = self.decoder4(x4)
        map4 = self.pred4(d4)

        # Stage 3
        d3 = self.decoder3(x3 + F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=False))
        map3 = self.pred3(d3)

        # Stage 2
        d2 = self.decoder2(x2 + F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False))
        map2 = self.pred2(d2)

        # Resize all to input resolution
        map2 = F.interpolate(map2, size=input_size, mode='bilinear', align_corners=False)
        map3 = F.interpolate(map3, size=input_size, mode='bilinear', align_corners=False)
        map4 = F.interpolate(map4, size=input_size, mode='bilinear', align_corners=False)
        coarse = F.interpolate(coarse, size=input_size, mode='bilinear', align_corners=False)

        return (
            map2,
            map3,
            map4,
            coarse,
            1 - torch.sigmoid(map2),
            1 - torch.sigmoid(map3),
            1 - torch.sigmoid(map4),
            1 - torch.sigmoid(coarse)
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


 
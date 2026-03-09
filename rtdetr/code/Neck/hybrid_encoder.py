import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Basic Conv -> BN -> Act
# =========================================================
class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# =========================================================
# Rep/CSP-like fusion block (bản đơn giản, dễ hiểu)
# Mục tiêu: trộn feature sau khi concat
# =========================================================
class CSPRepLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2, expansion=0.5):
        super().__init__()
        hidden = int(out_channels * expansion)

        self.conv1 = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.conv2 = ConvBNAct(in_channels, hidden, kernel_size=1)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    ConvBNAct(hidden, hidden, kernel_size=3),
                    ConvBNAct(hidden, hidden, kernel_size=3, act=False),
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.conv3 = ConvBNAct(hidden * 2, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        for block in self.blocks:
            residual = x1
            x1 = block(x1)
            x1 = x1 + residual

        out = torch.cat([x1, x2], dim=1)
        out = self.conv3(out)
        return out


# =========================================================
# 2D Positional Encoding (sin-cos) cho attention trên feature map
# =========================================================
def build_2d_sincos_pos_embed(h, w, dim, temperature=10000.0, device="cpu"):
    """
    Return: [1, H*W, C]
    """
    if dim % 4 != 0:
        raise ValueError(f"embed_dim phải chia hết cho 4, nhận được {dim}")

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij"
    )

    pos_dim = dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
    omega = 1.0 / (temperature ** omega)

    out_x = grid_x.reshape(-1, 1) * omega.reshape(1, -1)  # [HW, pos_dim]
    out_y = grid_y.reshape(-1, 1) * omega.reshape(1, -1)

    pos = torch.cat(
        [torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)],
        dim=1
    )  # [HW, dim]

    return pos.unsqueeze(0)  # [1, HW, dim]


# =========================================================
# Transformer Encoder Layer cho S5
# Intra-scale interaction
# =========================================================
class TransformerEncoderLayer2D(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ffn_dim=1024, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: [B, HW, C]
        """
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x = x + self.ffn(self.norm2(x))
        return x


# =========================================================
# Hybrid Encoder
# input : {"s3": [B,C3,H3,W3], "s4": [B,C4,H4,W4], "s5": [B,C5,H5,W5]}
# output: {"s3": [B,d,H3,W3],  "s4": [B,d,H4,W4],  "s5": [B,d,H5,W5]}
# =========================================================
class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels=(512, 1024, 2048),   # ví dụ cho ResNet50: s3,s4,s5
        hidden_dim=256,
        num_encoder_layers=1,
        num_heads=8,
        ffn_dim=1024,
        csp_blocks=2,
    ):
        super().__init__()

        c3, c4, c5 = in_channels
        self.hidden_dim = hidden_dim

        # -------------------------------------------------
        # 1) Project tất cả feature về cùng số channel
        # -------------------------------------------------
        self.input_proj_s3 = ConvBNAct(c3, hidden_dim, kernel_size=1)
        self.input_proj_s4 = ConvBNAct(c4, hidden_dim, kernel_size=1)
        self.input_proj_s5 = ConvBNAct(c5, hidden_dim, kernel_size=1)

        # -------------------------------------------------
        # 2) Intra-scale interaction trên S5 bằng self-attention
        # -------------------------------------------------
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer2D(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=0.0
            )
            for _ in range(num_encoder_layers)
        ])

        # -------------------------------------------------
        # 3) Top-down fusion
        #    s5 -> s4
        #    s4 -> s3
        # -------------------------------------------------
        self.topdown_fuse_s4 = CSPRepLayer(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            num_blocks=csp_blocks
        )
        self.topdown_fuse_s3 = CSPRepLayer(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            num_blocks=csp_blocks
        )

        # -------------------------------------------------
        # 4) Bottom-up fusion
        #    s3 -> s4
        #    s4 -> s5
        # -------------------------------------------------
        self.downsample_s3 = ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=2)
        self.bottomup_fuse_s4 = CSPRepLayer(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            num_blocks=csp_blocks
        )

        self.downsample_s4 = ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=2)
        self.bottomup_fuse_s5 = CSPRepLayer(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            num_blocks=csp_blocks
        )

    def forward(self, feats: dict):
        """
        feats = {
            "s3": feat3,
            "s4": feat4,
            "s5": feat5
        }
        """
        s3 = feats["s3"]
        s4 = feats["s4"]
        s5 = feats["s5"]

        # -------------------------------------------------
        # Step 1: unify channels
        # -------------------------------------------------
        s3 = self.input_proj_s3(s3)   # [B, d, H3, W3]
        s4 = self.input_proj_s4(s4)   # [B, d, H4, W4]
        s5 = self.input_proj_s5(s5)   # [B, d, H5, W5]

        # -------------------------------------------------
        # Step 2: intra-scale interaction trên S5
        # flatten -> attention -> reshape
        # -------------------------------------------------
        b, c, h, w = s5.shape
        pos = build_2d_sincos_pos_embed(h, w, c, device=s5.device)   # [1, HW, C]

        x = s5.flatten(2).transpose(1, 2)   # [B, HW, C]
        x = x + pos

        for layer in self.encoder_layers:
            x = layer(x)

        s5 = x.transpose(1, 2).reshape(b, c, h, w)

        # -------------------------------------------------
        # Step 3: top-down CCFF
        # s5 -> s4
        # -------------------------------------------------
        s5_up = F.interpolate(s5, size=s4.shape[-2:], mode="nearest")
        s4_td = self.topdown_fuse_s4(torch.cat([s4, s5_up], dim=1))

        # s4 -> s3
        s4_up = F.interpolate(s4_td, size=s3.shape[-2:], mode="nearest")
        s3_td = self.topdown_fuse_s3(torch.cat([s3, s4_up], dim=1))

        # -------------------------------------------------
        # Step 4: bottom-up CCFF
        # s3 -> s4
        # -------------------------------------------------
        s3_down = self.downsample_s3(s3_td)
        s4_out = self.bottomup_fuse_s4(torch.cat([s4_td, s3_down], dim=1))

        # s4 -> s5
        s4_down = self.downsample_s4(s4_out)
        s5_out = self.bottomup_fuse_s5(torch.cat([s5, s4_down], dim=1))

        return {
            "s3": s3_td,
            "s4": s4_out,
            "s5": s5_out,
        }


# =========================================================
# Quick test
# =========================================================
if __name__ == "__main__":
    feats = {
        "s3": torch.randn(2, 512, 80, 80),
        "s4": torch.randn(2, 1024, 40, 40),
        "s5": torch.randn(2, 2048, 20, 20),
    }

    neck = HybridEncoder(
        in_channels=(512, 1024, 2048),
        hidden_dim=256,
        num_encoder_layers=1,
        num_heads=8,
        ffn_dim=1024,
        csp_blocks=2,
    )

    outs = neck(feats)
    for k, v in outs.items():
        print(k, v.shape)
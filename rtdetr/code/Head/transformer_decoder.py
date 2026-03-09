import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Utility
# =========================================================
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x, inplace=True)
        return x


# =========================================================
# Positional embedding for query/reference
# =========================================================
class QueryPosEmbedding(nn.Module):
    """
    Convert reference points [B, N, 4] or [B, N, 2]
    into query positional embedding [B, N, C]
    """
    def __init__(self, input_dim=4, embed_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, ref_points):
        return self.proj(ref_points)


# =========================================================
# Flatten multi-scale features from neck
# input:
# {
#   "s3": [B, C, H3, W3],
#   "s4": [B, C, H4, W4],
#   "s5": [B, C, H5, W5]
# }
# output:
#   memory: [B, HW_total, C]
#   spatial_shapes: [num_levels, 2]
#   level_start_index: [num_levels]
# =========================================================
def flatten_multi_scale_feats(feats: dict):
    feat_list = [feats["s3"], feats["s4"], feats["s5"]]

    memories = []
    spatial_shapes = []

    for feat in feat_list:
        b, c, h, w = feat.shape
        memories.append(feat.flatten(2).transpose(1, 2))   # [B, HW, C]
        spatial_shapes.append([h, w])

    memory = torch.cat(memories, dim=1)  # [B, sum(HW), C]
    spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=memory.device)

    level_start_index = [0]
    for i in range(len(spatial_shapes) - 1):
        prev = spatial_shapes[i][0] * spatial_shapes[i][1]
        level_start_index.append(level_start_index[-1] + prev)
    level_start_index = torch.tensor(level_start_index, dtype=torch.long, device=memory.device)

    return memory, spatial_shapes, level_start_index


# =========================================================
# Placeholder Multi-Scale Deformable Attention
# Đây là bản KHUNG để bạn thay bằng bản thật sau.
# Hiện tại mình cho fallback về standard attention để code chạy được.
# =========================================================
class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(
        self,
        query,              # [B, Nq, C]
        memory,             # [B, Nm, C]
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        memory_key_padding_mask=None
    ):
        # Fallback implementation:
        # dùng standard cross-attention để giữ pipeline chạy được
        out, _ = self.attn(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask
        )
        return out


# =========================================================
# Decoder Layer
# =========================================================
class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.0,
        cross_attn_type="standard",  # "standard" | "deformable"
    ):
        super().__init__()

        self.cross_attn_type = cross_attn_type

        # self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # cross-attention: query attends to image memory
        if cross_attn_type == "deformable":
            self.cross_attn = MultiScaleDeformableAttention(
                embed_dim=embed_dim,
                num_heads=num_heads
            )
        else:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # FFN
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt,                        # [B, Nq, C]
        memory,                     # [B, Nm, C]
        query_pos=None,             # [B, Nq, C]
        reference_points=None,      # [B, Nq, 4] optional
        spatial_shapes=None,        # [L, 2]
        level_start_index=None,     # [L]
        memory_key_padding_mask=None
    ):
        # ------------------------------------------
        # 1) Self-attention
        # ------------------------------------------
        q = k = tgt if query_pos is None else (tgt + query_pos)

        tgt2, _ = self.self_attn(
            query=q,
            key=k,
            value=tgt
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ------------------------------------------
        # 2) Cross-attention
        # ------------------------------------------
        q = tgt if query_pos is None else (tgt + query_pos)

        if self.cross_attn_type == "deformable":
            tgt2 = self.cross_attn(
                query=q,
                memory=memory,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                memory_key_padding_mask=memory_key_padding_mask
            )
        else:
            tgt2, _ = self.cross_attn(
                query=q,
                key=memory,
                value=memory,
                key_padding_mask=memory_key_padding_mask
            )

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ------------------------------------------
        # 3) FFN
        # ------------------------------------------
        tgt2 = self.linear2(self.dropout(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


# =========================================================
# Transformer Decoder
# =========================================================
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.0,
        cross_attn_type="standard",  # "standard" or "deformable"
        learn_query_content=True,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # learnable query content
        if learn_query_content:
            self.query_content = nn.Embedding(num_queries, hidden_dim)
        else:
            self.query_content = None

        # learnable initial reference points
        self.query_refpoint = nn.Embedding(num_queries, 4)

        # query positional embedding from reference boxes
        self.query_pos_embed = QueryPosEmbedding(input_dim=4, embed_dim=hidden_dim)

        # decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                cross_attn_type=cross_attn_type
            )
            for _ in range(num_layers)
        ])

        # prediction heads for each layer
        self.class_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_layers)
        ])

        self.bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3)
            for _ in range(num_layers)
        ])

    def forward(
    self,
    tgt,
    ref_points,
    memory,
    spatial_shapes,
    level_start_index,
    attn_mask=None
):
    """
    tgt: [B, Q, C]
    ref_points: [B, Q, 4]
    memory: [B, N, C]
    """

    reference_points = ref_points

    outputs_classes = []
    outputs_boxes = []

    for lid, layer in enumerate(self.layers):
        query_pos = self.query_pos_embed(reference_points)

        tgt = layer(
            tgt=tgt,
            memory=memory,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            memory_key_padding_mask=None
        )

        out_class = self.class_head[lid](tgt)
        delta_bbox = self.bbox_head[lid](tgt)
        new_reference_points = (inverse_sigmoid(reference_points) + delta_bbox).sigmoid()

        outputs_classes.append(out_class)
        outputs_boxes.append(new_reference_points)

        reference_points = new_reference_points.detach()

    return {
        "pred_logits": outputs_classes[-1],
        "pred_boxes": outputs_boxes[-1],
        "aux_outputs": [
            {"pred_logits": cls, "pred_boxes": box}
            for cls, box in zip(outputs_classes[:-1], outputs_boxes[:-1])
        ]
    }


# =========================================================
# build function
# =========================================================
def build_transformer_decoder(cfg: dict):
    return TransformerDecoder(
        num_classes=cfg.get("num_classes", 80),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_queries=cfg.get("num_queries", 300),
        num_layers=cfg.get("num_layers", 6),
        num_heads=cfg.get("num_heads", 8),
        ffn_dim=cfg.get("ffn_dim", 1024),
        dropout=cfg.get("dropout", 0.0),
        cross_attn_type=cfg.get("cross_attn_type", "standard"),
        learn_query_content=cfg.get("learn_query_content", True),
    )


# =========================================================
# quick test
# =========================================================
if __name__ == "__main__":
    feats = {
        "s3": torch.randn(2, 256, 80, 80),
        "s4": torch.randn(2, 256, 40, 40),
        "s5": torch.randn(2, 256, 20, 20),
    }

    cfg = {
        "num_classes": 80,
        "hidden_dim": 256,
        "num_queries": 300,
        "num_layers": 6,
        "num_heads": 8,
        "ffn_dim": 1024,
        "dropout": 0.0,
        "cross_attn_type": "standard",  # đổi thành "deformable" để test khung deformable
    }

    decoder = build_transformer_decoder(cfg)
    outputs = decoder(feats)

    print("pred_logits:", outputs["pred_logits"].shape)
    print("pred_boxes :", outputs["pred_boxes"].shape)
    print("num aux    :", len(outputs["aux_outputs"]))
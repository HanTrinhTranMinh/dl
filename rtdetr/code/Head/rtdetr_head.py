import torch
import torch.nn as nn
import torch.nn.functional as F


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


def flatten_multi_scale_feats(feats: dict):
    """
    feats:
        {
            "s3": [B, C, H3, W3],
            "s4": [B, C, H4, W4],
            "s5": [B, C, H5, W5],
        }

    return:
        memory: [B, N, C]
        spatial_shapes: [L, 2]
        level_start_index: [L]
    """
    feat_list = [feats["s3"], feats["s4"], feats["s5"]]

    memories = []
    spatial_shapes = []

    for feat in feat_list:
        b, c, h, w = feat.shape
        memories.append(feat.flatten(2).transpose(1, 2))  # [B, HW, C]
        spatial_shapes.append([h, w])

    memory = torch.cat(memories, dim=1)
    spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=memory.device)

    level_start_index = [0]
    for i in range(len(spatial_shapes) - 1):
        prev = spatial_shapes[i][0] * spatial_shapes[i][1]
        level_start_index.append(level_start_index[-1] + prev)
    level_start_index = torch.tensor(level_start_index, dtype=torch.long, device=memory.device)

    return memory, spatial_shapes, level_start_index


class RTDETRHead(nn.Module):
    """
    Head nối Encoder -> Decoder

    Chức năng:
    1. Tạo encoder prediction sơ bộ trên flattened memory
    2. Query Selection: chọn top-k proposal tốt nhất
    3. Denoising Training: sinh noisy queries từ GT
    4. Ghép query cho decoder

    forward(...) trả về:
        {
            "decoder_inputs": {
                "tgt": ...,
                "ref_points": ...,
                "memory": ...,
                "spatial_shapes": ...,
                "level_start_index": ...
            },
            "enc_outputs": {
                "pred_logits": ...,
                "pred_boxes": ...
            },
            "dn_meta": ...
        }
    """
    def __init__(
        self,
        hidden_dim=256,
        num_classes=4,
        num_queries=300,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=0.4,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # encoder-side heads
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # project selected memory -> initial decoder tgt
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # label embedding for denoising queries
        self.label_embed = nn.Embedding(num_classes, hidden_dim)

    # --------------------------------------------------
    # Query Selection
    # --------------------------------------------------
    def select_topk_queries(self, memory, enc_logits, enc_boxes):
        """
        memory:     [B, N, C]
        enc_logits: [B, N, num_classes]
        enc_boxes:  [B, N, 4]

        chọn top-k theo max class score
        """
        bs, n, c = memory.shape

        scores = enc_logits.sigmoid().amax(dim=-1)       # [B, N]
        topk_scores, topk_indices = torch.topk(scores, k=self.num_queries, dim=1)

        topk_memory = torch.gather(
            memory,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, c)
        )  # [B, Q, C]

        topk_boxes = torch.gather(
            enc_boxes,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, 4)
        )  # [B, Q, 4]

        tgt = self.query_proj(topk_memory)
        ref_points = topk_boxes.sigmoid()

        return tgt, ref_points, topk_indices, topk_scores

    # --------------------------------------------------
    # Denoising Training
    # --------------------------------------------------
    def build_denoising_queries(self, targets, device):
        """
        targets: list[dict], mỗi phần tử:
            {
                "labels": Tensor[num_gt],
                "boxes":  Tensor[num_gt, 4]   # normalized cxcywh
            }

        return:
            dn_tgt:        [B, DN, C]
            dn_ref_points: [B, DN, 4]
            attn_mask:     optional
            dn_meta:       metadata
        """
        if targets is None or self.num_denoising <= 0:
            return None, None, None, None

        batch_size = len(targets)
        max_gt = max((len(t["labels"]) for t in targets), default=0)

        if max_gt == 0:
            return None, None, None, None

        # số nhóm lặp lại để đủ num_denoising
        num_group = max(1, self.num_denoising // max_gt)
        dn_number = num_group * max_gt

        dn_labels = torch.full(
            (batch_size, dn_number),
            fill_value=0,
            dtype=torch.long,
            device=device
        )
        dn_boxes = torch.zeros(batch_size, dn_number, 4, device=device)
        dn_valid_mask = torch.zeros(batch_size, dn_number, dtype=torch.bool, device=device)

        for b, tgt in enumerate(targets):
            labels = tgt["labels"].to(device)
            boxes = tgt["boxes"].to(device)

            num_gt = labels.shape[0]
            if num_gt == 0:
                continue

            # repeat nhiều lần để tạo denoising groups
            rep_labels = labels.repeat(num_group)
            rep_boxes = boxes.repeat(num_group, 1)

            total = min(dn_number, rep_labels.shape[0])
            rep_labels = rep_labels[:total]
            rep_boxes = rep_boxes[:total]

            # ----- label noise -----
            if self.label_noise_ratio > 0:
                noise_mask = torch.rand_like(rep_labels.float()) < self.label_noise_ratio
                random_labels = torch.randint(
                    low=0,
                    high=self.num_classes,
                    size=rep_labels.shape,
                    device=device
                )
                rep_labels = torch.where(noise_mask, random_labels, rep_labels)

            # ----- box noise -----
            if self.box_noise_scale > 0:
                noise = (torch.rand_like(rep_boxes) * 2.0 - 1.0) * self.box_noise_scale
                rep_boxes = (rep_boxes + noise).clamp(0.0, 1.0)

            dn_labels[b, :total] = rep_labels
            dn_boxes[b, :total] = rep_boxes
            dn_valid_mask[b, :total] = True

        dn_tgt = self.label_embed(dn_labels)          # [B, DN, C]
        dn_ref_points = dn_boxes                      # [B, DN, 4]

        dn_meta = {
            "dn_number": dn_number,
            "dn_valid_mask": dn_valid_mask,
            "num_group": num_group,
            "pad_size": dn_number,
        }

        return dn_tgt, dn_ref_points, None, dn_meta

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, feats, targets=None):
        """
        feats:
            {"s3": ..., "s4": ..., "s5": ...}

        targets:
            list[dict] hoặc None
            mỗi dict:
                {
                    "labels": Tensor[num_gt],
                    "boxes": Tensor[num_gt, 4]
                }
        """
        memory, spatial_shapes, level_start_index = flatten_multi_scale_feats(feats)
        # memory: [B, N, C]

        # encoder prediction sơ bộ trên từng token
        enc_logits = self.enc_score_head(memory)            # [B, N, num_classes]
        enc_boxes = self.enc_bbox_head(memory)              # [B, N, 4]

        # chọn top-k query tốt nhất
        tgt, ref_points, topk_indices, topk_scores = self.select_topk_queries(
            memory, enc_logits, enc_boxes
        )

        # denoising queries
        dn_tgt, dn_ref_points, attn_mask, dn_meta = self.build_denoising_queries(
            targets, device=memory.device
        )

        # nếu train có DN thì nối lên trước decoder queries
        if dn_tgt is not None:
            tgt = torch.cat([dn_tgt, tgt], dim=1)                  # [B, DN+Q, C]
            ref_points = torch.cat([dn_ref_points, ref_points], dim=1)  # [B, DN+Q, 4]

        return {
            "decoder_inputs": {
                "tgt": tgt,
                "ref_points": ref_points,
                "memory": memory,
                "spatial_shapes": spatial_shapes,
                "level_start_index": level_start_index,
                "attn_mask": attn_mask,
            },
            "enc_outputs": {
                "pred_logits": enc_logits,
                "pred_boxes": enc_boxes.sigmoid(),
                "topk_indices": topk_indices,
                "topk_scores": topk_scores,
            },
            "dn_meta": dn_meta,
        }


def build_rtdetr_head(cfg: dict):
    return RTDETRHead(
        hidden_dim=cfg.get("hidden_dim", 256),
        num_classes=cfg.get("num_classes", 4),
        num_queries=cfg.get("num_queries", 300),
        num_denoising=cfg.get("num_denoising", 100),
        label_noise_ratio=cfg.get("label_noise_ratio", 0.5),
        box_noise_scale=cfg.get("box_noise_scale", 0.4),
    )


if __name__ == "__main__":
    feats = {
        "s3": torch.randn(2, 256, 80, 80),
        "s4": torch.randn(2, 256, 40, 40),
        "s5": torch.randn(2, 256, 20, 20),
    }

    targets = [
        {
            "labels": torch.tensor([0, 1, 2]),
            "boxes": torch.tensor([
                [0.5, 0.5, 0.2, 0.2],
                [0.3, 0.4, 0.1, 0.15],
                [0.7, 0.6, 0.12, 0.10],
            ], dtype=torch.float32)
        },
        {
            "labels": torch.tensor([1, 3]),
            "boxes": torch.tensor([
                [0.2, 0.3, 0.1, 0.1],
                [0.8, 0.7, 0.2, 0.25],
            ], dtype=torch.float32)
        }
    ]

    cfg = {
        "hidden_dim": 256,
        "num_classes": 4,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 0.4,
    }

    head = build_rtdetr_head(cfg)
    out = head(feats, targets=targets)

    print("tgt:", out["decoder_inputs"]["tgt"].shape)
    print("ref_points:", out["decoder_inputs"]["ref_points"].shape)
    print("memory:", out["decoder_inputs"]["memory"].shape)
    print("enc_logits:", out["enc_outputs"]["pred_logits"].shape)
    print("enc_boxes:", out["enc_outputs"]["pred_boxes"].shape)
    print("dn_meta:", out["dn_meta"])
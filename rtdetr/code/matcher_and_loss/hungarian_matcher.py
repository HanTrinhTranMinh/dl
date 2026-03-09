import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes):
    """
    boxes: [N, 4] in (cx, cy, w, h)
    return: [N, 4] in (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_area(boxes):
    """
    boxes: [N, 4] in xyxy
    """
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4]
    boxes2: [M, 4]
    return:
        iou:   [N, M]
        union: [N, M]
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # [N, M, 2]

    wh = (rb - lt).clamp(min=0)                          # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]                   # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-7)

    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4] xyxy
    boxes2: [M, 4] xyxy
    return:
        giou: [N, M]
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])   # enclosing top-left
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])   # enclosing bottom-right

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area - union) / (area + 1e-7)
    return giou


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for DETR / RT-DETR style matching.

    Cost = cost_class + cost_bbox + cost_giou

    Args:
        cost_class: weight for classification cost
        cost_bbox:  weight for bbox L1 cost
        cost_giou:  weight for giou cost
        use_focal_loss: nếu True thì dùng focal-style classification cost
    """
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        use_focal_loss: bool = False,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, \
            "All costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs:
            {
                "pred_logits": [B, Q, num_classes],
                "pred_boxes":  [B, Q, 4]   # cxcywh normalized
            }

        targets: list of dict
            [
                {
                    "labels": Tensor[num_gt],
                    "boxes":  Tensor[num_gt, 4]   # cxcywh normalized
                },
                ...
            ]

        return:
            indices: list of tuples
                [
                    (index_i, index_j),
                    ...
                ]
            với:
                index_i: indices dự đoán được match
                index_j: indices ground-truth tương ứng
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_logits = outputs["pred_logits"]   # [B, Q, C]
        out_bbox = outputs["pred_boxes"]      # [B, Q, 4]

        indices = []

        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]

            # không có GT
            if tgt_ids.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64)
                ))
                continue

            # --------------------------------------------------
            # 1) Classification cost
            # --------------------------------------------------
            if self.use_focal_loss:
                # Focal-style classification cost
                out_prob = out_logits[b].sigmoid()  # [Q, C]

                neg_cost = (1 - self.alpha) * (out_prob ** self.gamma) * (
                    -(1 - out_prob + 1e-8).log()
                )
                pos_cost = self.alpha * ((1 - out_prob) ** self.gamma) * (
                    -(out_prob + 1e-8).log()
                )

                cost_class = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]  # [Q, num_gt]
            else:
                # Softmax CE-style cost
                out_prob = out_logits[b].softmax(-1)   # [Q, C]
                cost_class = -out_prob[:, tgt_ids]     # [Q, num_gt]

            # --------------------------------------------------
            # 2) BBox L1 cost
            # --------------------------------------------------
            # torch.cdist with p=1 => pairwise L1 distance
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)   # [Q, num_gt]

            # --------------------------------------------------
            # 3) GIoU cost
            # --------------------------------------------------
            out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox[b])
            tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)

            cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)   # [Q, num_gt]

            # --------------------------------------------------
            # Final cost matrix
            # --------------------------------------------------
            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )  # [Q, num_gt]

            C = C.cpu()

            pred_ind, tgt_ind = linear_sum_assignment(C)

            indices.append((
                torch.as_tensor(pred_ind, dtype=torch.int64),
                torch.as_tensor(tgt_ind, dtype=torch.int64)
            ))

        return indices


def build_hungarian_matcher(cfg: dict):
    return HungarianMatcher(
        cost_class=cfg.get("cost_class", 2.0),
        cost_bbox=cfg.get("cost_bbox", 5.0),
        cost_giou=cfg.get("cost_giou", 2.0),
        use_focal_loss=cfg.get("use_focal_loss", False),
        alpha=cfg.get("alpha", 0.25),
        gamma=cfg.get("gamma", 2.0),
    )


if __name__ == "__main__":
    outputs = {
        "pred_logits": torch.randn(2, 300, 4),   # B=2, Q=300, 4 classes
        "pred_boxes": torch.rand(2, 300, 4),     # normalized cxcywh
    }

    targets = [
        {
            "labels": torch.tensor([0, 1, 2], dtype=torch.int64),
            "boxes": torch.tensor([
                [0.5, 0.5, 0.2, 0.2],
                [0.3, 0.4, 0.1, 0.15],
                [0.7, 0.6, 0.12, 0.10],
            ], dtype=torch.float32)
        },
        {
            "labels": torch.tensor([1, 3], dtype=torch.int64),
            "boxes": torch.tensor([
                [0.2, 0.3, 0.1, 0.1],
                [0.8, 0.7, 0.2, 0.25],
            ], dtype=torch.float32)
        }
    ]

    cfg = {
        "cost_class": 2.0,
        "cost_bbox": 5.0,
        "cost_giou": 2.0,
        "use_focal_loss": False,
    }

    matcher = build_hungarian_matcher(cfg)
    indices = matcher(outputs, targets)

    for i, (src_idx, tgt_idx) in enumerate(indices):
        print(f"Batch {i}")
        print(" matched pred idx:", src_idx)
        print(" matched gt   idx:", tgt_idx)
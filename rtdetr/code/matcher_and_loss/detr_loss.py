import torch
import torch.nn as nn
import torch.nn.functional as F


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
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4]
    boxes2: [M, 4]
    return:
        iou:   [N, M]
        union: [N, M]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # [N, M, 2]

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-7)

    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4] xyxy
    boxes2: [N, 4] xyxy
    return:
        giou: [N]
    """
    assert boxes1.shape == boxes2.shape

    iou, union = box_iou(boxes1, boxes2)
    iou = iou.diag()
    union = union.diag()

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, 0] * wh[:, 1]

    giou = iou - (area - union) / (area + 1e-7)
    return giou


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss cho multi-label classification.

    pred_logits: [B, Q, C]
    target_score: [B, Q, C]  (IoU-aware target score)
    label: [B, Q, C]         (0/1)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target_score, label):
        pred_sigmoid = pred_logits.sigmoid()

        weight = self.alpha * pred_sigmoid.pow(self.gamma) * (1 - label) + target_score * label

        loss = F.binary_cross_entropy_with_logits(
            pred_logits,
            target_score,
            reduction="none"
        )

        loss = loss * weight
        return loss


class DETRLoss(nn.Module):
    """
    Loss cho DETR / RT-DETR style.

    Bao gồm:
    - Varifocal Loss cho classification
    - L1 Loss cho box
    - GIoU Loss cho box
    - Auxiliary losses cho các layer trung gian
    """
    def __init__(
        self,
        matcher,
        num_classes=4,
        weight_dict=None,
        alpha=0.75,
        gamma=2.0,
        aux_loss=True,
    ):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.aux_loss = aux_loss

        self.vfl = VarifocalLoss(alpha=alpha, gamma=gamma)

        if weight_dict is None:
            weight_dict = {
                "loss_vfl": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
            }
        self.weight_dict = weight_dict

    # --------------------------------------------------
    # utils
    # --------------------------------------------------
    def _get_src_permutation_idx(self, indices):
        """
        indices: list[(src_idx, tgt_idx)] for each batch item
        return flattened batch_idx and src_idx
        """
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_num_boxes(self, targets, device):
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        num_boxes = torch.clamp(num_boxes, min=1.0).item()
        return num_boxes

    def _build_target_classes_and_boxes(self, outputs, targets, indices):
        """
        outputs:
            pred_logits: [B, Q, C]
            pred_boxes:  [B, Q, 4]

        return:
            target_labels_full: [B, Q, C]
            target_scores_full: [B, Q, C]
            matched_pred_boxes: [M, 4]
            matched_tgt_boxes:  [M, 4]
        """
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        bs, num_queries, num_classes = pred_logits.shape
        device = pred_logits.device

        target_labels_full = torch.zeros((bs, num_queries, num_classes), device=device)
        target_scores_full = torch.zeros((bs, num_queries, num_classes), device=device)

        matched_pred_boxes = []
        matched_tgt_boxes = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            tgt_labels = targets[b]["labels"][tgt_idx]       # [M]
            tgt_boxes = targets[b]["boxes"][tgt_idx]         # [M, 4]
            src_boxes = pred_boxes[b][src_idx]               # [M, 4]

            # one-hot label
            target_labels_full[b, src_idx, tgt_labels] = 1.0

            # IoU-aware target score cho VFL
            src_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            ious = generalized_box_iou(src_xyxy, tgt_xyxy).clamp(min=0.0)  # [M]

            target_scores_full[b, src_idx, tgt_labels] = ious

            matched_pred_boxes.append(src_boxes)
            matched_tgt_boxes.append(tgt_boxes)

        if len(matched_pred_boxes) > 0:
            matched_pred_boxes = torch.cat(matched_pred_boxes, dim=0)
            matched_tgt_boxes = torch.cat(matched_tgt_boxes, dim=0)
        else:
            matched_pred_boxes = torch.zeros((0, 4), device=device)
            matched_tgt_boxes = torch.zeros((0, 4), device=device)

        return target_labels_full, target_scores_full, matched_pred_boxes, matched_tgt_boxes

    # --------------------------------------------------
    # classification loss
    # --------------------------------------------------
    def loss_labels(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs["pred_logits"]  # [B, Q, C]

        target_labels_full, target_scores_full, _, _ = self._build_target_classes_and_boxes(
            outputs, targets, indices
        )

        loss_vfl = self.vfl(pred_logits, target_scores_full, target_labels_full)
        loss_vfl = loss_vfl.sum() / num_boxes

        return {"loss_vfl": loss_vfl}

    # --------------------------------------------------
    # bbox losses
    # --------------------------------------------------
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        _, _, matched_pred_boxes, matched_tgt_boxes = self._build_target_classes_and_boxes(
            outputs, targets, indices
        )

        if matched_pred_boxes.numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {
                "loss_bbox": zero,
                "loss_giou": zero,
            }

        loss_bbox = F.l1_loss(matched_pred_boxes, matched_tgt_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        pred_xyxy = box_cxcywh_to_xyxy(matched_pred_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(matched_tgt_boxes)

        giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
        loss_giou = (1.0 - giou).sum() / num_boxes

        return {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }

    # --------------------------------------------------
    # compute loss for one output dict
    # --------------------------------------------------
    def _compute_single(self, outputs, targets, suffix=""):
        indices = self.matcher(outputs, targets)
        num_boxes = self._get_num_boxes(targets, device=outputs["pred_logits"].device)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        if suffix:
            losses = {k + suffix: v for k, v in losses.items()}

        return losses

    # --------------------------------------------------
    # forward
    # --------------------------------------------------
    def forward(self, outputs, targets):
        """
        outputs:
            {
                "pred_logits": [B, Q, C],
                "pred_boxes": [B, Q, 4],
                "aux_outputs": [
                    {"pred_logits": ..., "pred_boxes": ...},
                    ...
                ]  # optional
            }

        targets:
            list[dict]
            [
                {
                    "labels": Tensor[num_gt],
                    "boxes": Tensor[num_gt, 4]
                },
                ...
            ]
        """
        losses = {}

        # main output
        main_losses = self._compute_single(
            {
                "pred_logits": outputs["pred_logits"],
                "pred_boxes": outputs["pred_boxes"],
            },
            targets,
            suffix=""
        )
        losses.update(main_losses)

        # auxiliary outputs
        if self.aux_loss and "aux_outputs" in outputs:
            for i, aux_out in enumerate(outputs["aux_outputs"]):
                aux_losses = self._compute_single(
                    {
                        "pred_logits": aux_out["pred_logits"],
                        "pred_boxes": aux_out["pred_boxes"],
                    },
                    targets,
                    suffix=f"_aux_{i}"
                )
                losses.update(aux_losses)

        # weighted total
        total_loss = 0.0
        for k, v in losses.items():
            base_key = k
            if "_aux_" in k:
                base_key = k.split("_aux_")[0]
            total_loss = total_loss + self.weight_dict.get(base_key, 1.0) * v

        losses["loss_total"] = total_loss
        return losses


def build_detr_loss(cfg: dict, matcher):
    return DETRLoss(
        matcher=matcher,
        num_classes=cfg.get("num_classes", 4),
        weight_dict=cfg.get(
            "weight_dict",
            {
                "loss_vfl": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
            }
        ),
        alpha=cfg.get("vfl_alpha", 0.75),
        gamma=cfg.get("vfl_gamma", 2.0),
        aux_loss=cfg.get("aux_loss", True),
    )


if __name__ == "__main__":
    # cần matcher từ hungarian_matcher.py
    from hungarian_matcher import HungarianMatcher

    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        use_focal_loss=False
    )

    criterion = DETRLoss(
        matcher=matcher,
        num_classes=4,
        weight_dict={
            "loss_vfl": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        alpha=0.75,
        gamma=2.0,
        aux_loss=True,
    )

    outputs = {
        "pred_logits": torch.randn(2, 300, 4),
        "pred_boxes": torch.rand(2, 300, 4),
        "aux_outputs": [
            {
                "pred_logits": torch.randn(2, 300, 4),
                "pred_boxes": torch.rand(2, 300, 4),
            },
            {
                "pred_logits": torch.randn(2, 300, 4),
                "pred_boxes": torch.rand(2, 300, 4),
            }
        ]
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

    losses = criterion(outputs, targets)
    for k, v in losses.items():
        print(k, float(v.detach()))
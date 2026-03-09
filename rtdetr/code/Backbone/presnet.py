import torch
import torch.nn as nn

from bottleneck import BasicBlock, Bottleneck


class ConvBNAct(nn.Module):
    """
    Conv -> BN -> ReLU
    Dùng cho stem của PResNet.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class PResNet(nn.Module):
    """
    PResNet backbone.

    Forward output:
        {
            "s3": feat3,
            "s4": feat4,
            "s5": feat5
        }

    Config ví dụ:
        cfg = {
            "depth": 50,
            "variant": "d",
            "num_stages": 4,
            "return_idx": [1, 2, 3],   # tương ứng layer2, layer3, layer4
            "freeze_at": -1,
            "freeze_norm": False,
            "in_channels": 3
        }
    """
    def __init__(self, cfg: dict):
        super().__init__()

        depth = cfg.get("depth", 50)
        variant = cfg.get("variant", "d")
        num_stages = cfg.get("num_stages", 4)
        return_idx = cfg.get("return_idx", [1, 2, 3])
        freeze_at = cfg.get("freeze_at", -1)
        freeze_norm = cfg.get("freeze_norm", False)
        in_channels = cfg.get("in_channels", 3)

        assert depth in [18, 34, 50, 101], f"Unsupported depth: {depth}"
        assert variant in ["b", "d"], f"Unsupported variant: {variant}"
        assert 1 <= num_stages <= 4, f"num_stages must be in [1,4], got {num_stages}"

        self.depth = depth
        self.variant = variant
        self.num_stages = num_stages
        self.return_idx = return_idx
        self.freeze_at = freeze_at
        self.freeze_norm = freeze_norm

        # --------------------------------------------------
        # Chọn block type và số block mỗi stage theo depth
        # --------------------------------------------------
        if depth == 18:
            block = BasicBlock
            stage_blocks = [2, 2, 2, 2]
            stage_channels = [64, 128, 256, 512]
        elif depth == 34:
            block = BasicBlock
            stage_blocks = [3, 4, 6, 3]
            stage_channels = [64, 128, 256, 512]
        elif depth == 50:
            block = Bottleneck
            stage_blocks = [3, 4, 6, 3]
            stage_channels = [64, 128, 256, 512]
        elif depth == 101:
            block = Bottleneck
            stage_blocks = [3, 4, 23, 3]
            stage_channels = [64, 128, 256, 512]

        self.block = block
        self.stage_blocks = stage_blocks
        self.stage_channels = stage_channels
        self.out_channels = [c * block.expansion for c in stage_channels]

        # current inplanes sau stem
        self.inplanes = 64

        # --------------------------------------------------
        # Stem
        # variant d: 3 conv 3x3
        # variant b: 7x7 conv chuẩn
        # --------------------------------------------------
        if variant == "d":
            self.stem = nn.Sequential(
                ConvBNAct(in_channels, 32, kernel_size=3, stride=2, padding=1),
                ConvBNAct(32, 32, kernel_size=3, stride=1, padding=1),
                ConvBNAct(32, 64, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.stem = nn.Sequential(
                ConvBNAct(in_channels, 64, kernel_size=7, stride=2, padding=3),
            )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --------------------------------------------------
        # Stages
        # layer1 -> layer4
        # --------------------------------------------------
        self.res_layers = nn.ModuleList()

        for i in range(num_stages):
            out_channels = stage_channels[i]
            num_blocks = stage_blocks[i]
            stride = 1 if i == 0 else 2
            layer = self._make_layer(
                block=block,
                out_channels=out_channels,
                blocks=num_blocks,
                stride=stride
            )
            self.res_layers.append(layer)

        # init weights
        self._init_weights()

        # freeze nếu cần
        if self.freeze_norm:
            self._freeze_norm()

        if self.freeze_at >= 0:
            self._freeze_parameters(self.freeze_at)

    def _make_downsample(self, out_channels, stride):
        """
        Downsample cho shortcut branch.
        Variant d dùng AvgPool2d + Conv1x1 + BN khi stride=2.
        """
        out_expanded = out_channels * self.block.expansion

        if self.variant == "d" and stride == 2:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
                nn.Conv2d(
                    self.inplanes,
                    out_expanded,
                    kernel_size=1,
                    stride=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_expanded)
            )
        else:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    out_expanded,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_expanded)
            )

        return downsample

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Tạo 1 stage gồm nhiều residual blocks.
        """
        downsample = None
        out_expanded = out_channels * block.expansion

        if stride != 1 or self.inplanes != out_expanded:
            downsample = self._make_downsample(out_channels, stride)

        layers = []
        layers.append(block(self.inplanes, out_channels, stride=stride, downsample=downsample))
        self.inplanes = out_expanded

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _freeze_norm(self):
        """
        Freeze toàn bộ BatchNorm:
        - eval mode
        - không cập nhật weight/bias
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def _freeze_parameters(self, freeze_at: int):
        """
        freeze_at:
            0 -> freeze stem
            1 -> freeze stem + layer1
            2 -> freeze stem + layer1 + layer2
            ...
        """
        if freeze_at >= 0:
            for p in self.stem.parameters():
                p.requires_grad = False

        for i in range(min(freeze_at, len(self.res_layers))):
            for p in self.res_layers[i].parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        Output:
            {
                "s3": feat3,
                "s4": feat4,
                "s5": feat5
            }

        Mapping:
            layer1 -> C2
            layer2 -> C3 -> s3
            layer3 -> C4 -> s4
            layer4 -> C5 -> s5
        """
        x = self.stem(x)       # /2
        x = self.maxpool(x)    # /4

        outs = []

        for i, layer in enumerate(self.res_layers):
            x = layer(x)
            outs.append(x)

        # mặc định return_idx = [1,2,3] => layer2, layer3, layer4
        selected = [outs[i] for i in self.return_idx]

        assert len(selected) == 3, (
            f"PResNet forward expects 3 output features for s3/s4/s5, "
            f"but got {len(selected)} from return_idx={self.return_idx}"
        )

        return {
            "s3": selected[0],
            "s4": selected[1],
            "s5": selected[2],
        }


# ----------------------------------------------------------
# Factory function
# ----------------------------------------------------------
def build_presnet(cfg: dict):
    return PResNet(cfg)


# ----------------------------------------------------------
# Quick test
# ----------------------------------------------------------
if __name__ == "__main__":
    cfg = {
        "depth": 50,
        "variant": "d",
        "num_stages": 4,
        "return_idx": [1, 2, 3],   # layer2, layer3, layer4
        "freeze_at": -1,
        "freeze_norm": False,
        "in_channels": 3,
    }

    model = build_presnet(cfg)
    x = torch.randn(1, 3, 640, 640)

    feats = model(x)
    for k, v in feats.items():
        print(k, v.shape)
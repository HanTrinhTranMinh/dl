# resnet.py: Định nghĩa lớp cha (Base Class).
# Tạo khung xương chung cho ResNet, quản lý các stage (layer1 đến layer4).

import torch
import torch.nn as nn


# =========================================================
# BasicBlock
# Dùng cho ResNet18 / ResNet34
# =========================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)   # Conv -> BN -> ReLU
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # Conv -> BN
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


# =========================================================
# Bottleneck
# Dùng cho ResNet50 / 101 / 152
# =========================================================
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)   # 1x1
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # 1x1
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


# =========================================================
# ResNet Base Class
# Tạo khung xương chung cho backbone
# =========================================================
class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        in_channels=3,
        num_classes=1000,
        variant="b",         # "b" = chuẩn, "d" = ResNet-D
        return_stages=False  # True: trả về feature maps layer1..4
    ):
        super().__init__()

        self.block = block
        self.layers = layers
        self.variant = variant
        self.return_stages = return_stages

        # current number of channels sau stem
        self.inplanes = 64

        # -------------------------------------------------
        # Stem
        # -------------------------------------------------
        if variant == "d":
            # ResNet-D stem: 3 conv 3x3
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        else:
            # ResNet chuẩn
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # -------------------------------------------------
        # 4 stages
        # -------------------------------------------------
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # -------------------------------------------------
        # Classification head
        # -------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    # -----------------------------------------------------
    # Tạo downsample cho skip connection
    # -----------------------------------------------------
    def _make_downsample(self, out_channels, stride):
        out_expanded = out_channels * self.block.expansion

        if self.variant == "d" and stride == 2:
            # ResNet-D: AvgPool -> Conv1x1 -> BN
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
                nn.Conv2d(self.inplanes, out_expanded, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_expanded),
            )
        else:
            # ResNet chuẩn
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_expanded, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_expanded),
            )

        return downsample

    # -----------------------------------------------------
    # Tạo 1 stage gồm nhiều block
    # -----------------------------------------------------
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        out_expanded = out_channels * block.expansion

        # Nếu shape đầu vào != shape đầu ra thì phải downsample identity
        if stride != 1 or self.inplanes != out_expanded:
            downsample = self._make_downsample(out_channels, stride)

        layers = []
        # block đầu tiên của stage có thể downsample
        layers.append(block(self.inplanes, out_channels, stride=stride, downsample=downsample))
        self.inplanes = out_expanded

        # các block còn lại giữ nguyên shape
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, stride=1, downsample=None))

        return nn.Sequential(*layers)

    # -----------------------------------------------------
    # Khởi tạo trọng số
    # -----------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------
    def forward(self, x):
        x = self.stem(x)     # /2
        x = self.maxpool(x)  # /4

        c2 = self.layer1(x)  # thường gọi là C2
        c3 = self.layer2(c2) # C3
        c4 = self.layer3(c3) # C4
        c5 = self.layer4(c4) # C5

        if self.return_stages:
            return [c2, c3, c4, c5]

        x = self.avgpool(c5)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =========================================================
# Các hàm dựng model nhanh
# =========================================================
def resnet18(variant="b", return_stages=False, num_classes=1000):
    return ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        variant=variant,
        return_stages=return_stages,
        num_classes=num_classes
    )


def resnet34(variant="b", return_stages=False, num_classes=1000):
    return ResNet(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        variant=variant,
        return_stages=return_stages,
        num_classes=num_classes
    )


def resnet50(variant="b", return_stages=False, num_classes=1000):
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        variant=variant,
        return_stages=return_stages,
        num_classes=num_classes
    )


def resnet101(variant="b", return_stages=False, num_classes=1000):
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        variant=variant,
        return_stages=return_stages,
        num_classes=num_classes
    )


# =========================================================
# Test nhanh
# =========================================================
if __name__ == "__main__":
    x = torch.randn(1, 3, 640, 640)

    model = resnet50(variant="d", return_stages=True)
    feats = model(x)

    for i, f in enumerate(feats, start=2):
        print(f"C{i}: {f.shape}")
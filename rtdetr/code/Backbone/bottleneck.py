# Ở đây sẽ là CNN (mặc định là PResNet) trích xuất đặc trưng đa tỷ lệ

import torch
import torch.nn as nn
import torchvision.models as models
import torchinfo

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module): # Bottleneck là một block cơ bản trong kiến trúc ResNet, được thiết kế để giảm số lượng tham số và tăng hiệu quả tính toán trong các mạng sâu. Nó sử dụng một cấu trúc ba lớp convolution để trích xuất đặc trưng, với một đường tắt (shortcut) để giúp truyền thông tin qua các lớp mà không bị mất mát.
    expansion = 4 #Số lượng kênh đầu ra sẽ là out_channels * expansion

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        #Conv1 giảm số lượng kênh từ in_channels xuống out_channels (HxW vẫn giữ nguyên)
        self.bn1 = nn.BatchNorm2d(out_channels)
        #BachNorm sau mỗi convolution để ổn định quá trình huấn luyện
        #BatchNorm 2d (CNN) mean = 0, variance = 1, [B, C, H, W]
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        #Conv2 là convolution chính, giữ nguyên số lượng kênh nhưng có thể thay đổi kích thước không gian nếu stride > 1
        self.bn2 = nn.BatchNorm2d(out_channels)
        #BatchNorm sau conv2 để ổn định quá trình huấn luyện
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        #Conv3 tăng số lượng kênh từ out_channels lên out_channels * expansion (HxW vẫn giữ nguyên)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        #BatchNorm sau conv3 để ổn định quá trình huấn luyện
        self.relu = nn.ReLU(inplace=True)
        #ReLU là hàm kích hoạt phi tuyến, inplace=True để tiết kiệm bộ nhớ bằng cách thực hiện phép toán tại chỗ
        self.downsample = downsample
        #Downsample là một module tùy chọn để điều chỉnh kích thước của identity nếu cần thiết (khi stride > 1 hoặc số kênh thay đổi)

    def forward(self, x):
        identity = x

        out = self.conv1(x) #Conv -> BatchNorm -> ReLU
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)#Conv -> BatchNorm -> ReLU
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) #Conv -> BatchNorm (không có ReLU sau conv3 vì sẽ cộng với identity trước khi kích hoạt)
        out = self.bn3(out)

        if self.downsample is not None: 
            identity = self.downsample(x)

        out += identity #Cộng phần đầu vào (identity) với phần đầu ra của conv3
        out = self.relu(out) #Kích hoạt ReLU sau khi cộng để tạo ra đầu ra cuối cùng của block

        return out
    

class BottleneckD(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:

            layers = []

            if stride == 2:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))

            layers.append(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=1, bias=False)
            )
            layers.append(
                nn.BatchNorm2d(out_channels * self.expansion)
            )

            self.downsample = nn.Sequential(*layers)

    def forward(self, x):

        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)

        return out
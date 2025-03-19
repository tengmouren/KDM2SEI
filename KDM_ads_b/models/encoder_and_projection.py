# import torchvision.models as models
import torch
from .mlp_head import MLPHead
from torch import nn
import torch.nn.functional as F

from .complexcnn import ComplexConv, ComplexConv_trans


class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        out = self.avg_pool(x).squeeze(-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = out.unsqueeze(-1)
        return x * out


class Encoder_and_projection(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Encoder_and_projection, self).__init__()
        self.conv1 = ComplexConv(
            in_channels=1, out_channels=64, kernel_size=4, stride=2
        )
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.se1 = SqueezeExciteBlock(128)  # 添加SENet模块
        self.conv2 = ComplexConv(
            in_channels=64, out_channels=64, kernel_size=4, stride=2
        )
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.se2 = SqueezeExciteBlock(128)  # 添加SENet模块
        self.conv3 = ComplexConv(
            in_channels=64, out_channels=64, kernel_size=4, stride=2
        )
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.se3 = SqueezeExciteBlock(128)  # 添加SENet模块
        self.conv4 = ComplexConv(
            in_channels=64, out_channels=64, kernel_size=4, stride=2
        )
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.se4 = SqueezeExciteBlock(128)  # 添加SENet模块
        self.conv5 = ComplexConv(
            in_channels=64, out_channels=64, kernel_size=4, stride=2
        )
        self.bn5 = nn.BatchNorm1d(num_features=128)
        self.se5 = SqueezeExciteBlock(128)  # 添加SENet模块
        self.conv6 = ComplexConv(
            in_channels=64, out_channels=64, kernel_size=4, stride=2
        )
        self.bn6 = nn.BatchNorm1d(num_features=128)
        self.se6 = SqueezeExciteBlock(128)  # 添加SENet模块
        self.conv7 = ComplexConv(
            in_channels=64, out_channels=64, kernel_size=4, stride=2
        )
        self.bn7 = nn.BatchNorm1d(num_features=128)
        self.se7 = SqueezeExciteBlock(128)  # 添加SENet模块
        self.conv8 = ComplexConv(
            in_channels=64, out_channels=64, kernel_size=4, stride=2
        )
        self.bn8 = nn.BatchNorm1d(num_features=128)
        self.se8 = SqueezeExciteBlock(128)  # 添加SENet模块
        self.conv9 = ComplexConv(
            in_channels=64, out_channels=64, kernel_size=4, stride=2
        )
        self.bn9 = nn.BatchNorm1d(num_features=128)
        self.se9 = SqueezeExciteBlock(128)  # 添加SENet模块
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(1024)
        # 512 128
        self.projetion = MLPHead(in_channels=1024, **kwargs["projection_head"])

    def forward(self, x):
        # 将后28维拼接到最后的输出
        x_input = x[:, :, :4800]
        x_extra = x[:, 0, -20:]

        # 维度变化：(batch_size, 2, input_length) -> (batch_size, 128, new_length)
        x = self.conv1(x_input)
        x = F.relu(x)  # relu激活函数
        x = self.bn1(x)  # 批归一化
        x = self.se1(x)  # SENet模块增强表征能力

        # 维度变化：(batch_size, 128, new_length) -> (batch_size, 128, new_length_2)
        x = self.conv2(x)
        x = F.relu(x)  # relu激活函数
        x = self.bn2(x)  # 批归一化
        x = self.se2(x)  # SENet模块增强表征能力

        # 维度变化：(batch_size, 128, new_length_2) -> (batch_size, 128, new_length_3)
        x = self.conv3(x)
        x = F.relu(x)  # relu激活函数
        x = self.bn3(x)  # 批归一化
        x = self.se3(x)  # SENet模块增强表征能力

        # 维度变化：(batch_size, 128, new_length_3) -> (batch_size, 128, new_length_4)
        x = self.conv4(x)
        x = F.relu(x)  # relu激活函数
        x = self.bn4(x)  # 批归一化
        x = self.se4(x)  # SENet模块增强表征能力

        # 维度变化：(batch_size, 128, new_length_4) -> (batch_size, 128, new_length_5)
        x = self.conv5(x)
        x = F.relu(x)  # relu激活函数
        x = self.bn5(x)  # 批归一化
        x = self.se5(x)  # SENet模块增强表征能力

        # 维度变化：(batch_size, 128, new_length_5) -> (batch_size, 128, new_length_6)
        x = self.conv6(x)
        x = F.relu(x)  # relu激活函数
        x = self.bn6(x)  # 批归一化
        x = self.se6(x)  # SENet模块增强表征能力

        # 维度变化：(batch_size, 128, new_length_6) -> (batch_size, 128, new_length_7)
        x = self.conv7(x)
        x = F.relu(x)  # relu激活函数
        x = self.bn7(x)  # 批归一化
        x = self.se7(x)  # SENet模块增强表征能力

        # 维度变化：(batch_size, 128, new_length_7) -> (batch_size, 128, new_length_8)
        x = self.conv8(x)
        x = F.relu(x)  # relu激活函数
        x = self.bn8(x)  # 批归一化
        x = self.se8(x)  # SENet模块增强表征能力

        # 维度变化：(batch_size, 128, new_length_8) -> (batch_size, 128, new_length_9)
        x = self.conv9(x)
        x = F.relu(x)  # relu激活函数
        x = self.bn9(x)  # 批归一化
        x = self.se9(x)  # SENet模块增强表征能力

        # 展平，维度变化：(batch_size, 128, new_length_9) -> (batch_size, 128 * new_length_9)
        x = self.flatten(x)

        # 全连接层，维度变化：(batch_size, 128 * new_length_9) -> (batch_size, 1024)
        x = self.fc(x)
        embedding = F.relu(x)  # relu激活函数
        # 投影层，维度变化：(batch_size, 1024) -> (batch_size, 1024+20=1044)
        embedding_new = torch.cat((embedding, x_extra), dim=1)
        #  (batch_size, 1024) ->> (batch_size, 512)->128
        project_out = self.projetion(embedding)
        if 1 == 0:
            # 预训练
            return embedding, project_out
        else:
            # 微调
            return embedding, embedding_new

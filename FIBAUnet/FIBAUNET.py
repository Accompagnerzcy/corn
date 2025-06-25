import torch
import torch.nn as nn
import timm
from fvcore.nn import FlopCountAnalysis, parameter_count


class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False, groups=1):
        super(Conv2dStaticSamePadding, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.groups = groups
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=bias, padding=0)
        # Calculate padding
        self.padding = self._get_padding(kernel_size, stride)
        self.pad = nn.ZeroPad2d(self.padding)

    def _get_padding(self, kernel_size, stride):
        # Assuming kernel_size and stride are tuples of 2
        padding = [(k - 1) // 2 for k in kernel_size]
        return padding[1], padding[1], padding[0], padding[0]

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25):
        #0.25作为缩小参数是经验值，检查了所有的efficientent-b3中的se_ratio都是0.25
        super(MBConvBlock, self).__init__()
        mid_channels = in_channels * expand_ratio

        # Expand phase
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.010000000000000009)

        # Depthwise convolution
        self.depthwise_conv = Conv2dStaticSamePadding(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, groups=mid_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.010000000000000009)

        # Squeeze and Excitation
        se_mid = max(1, int(in_channels * se_ratio))
        self.se_reduce = nn.Conv2d(mid_channels, se_mid, kernel_size=1, stride=1)
        self.se_expand = nn.Conv2d(se_mid, mid_channels, kernel_size=1, stride=1)

        # Output phase
        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.010000000000000009)

        self.swish = nn.SiLU()  # MemoryEfficientSwish is an optimized version of SiLU

    def forward(self, x):
        residual = x

        # Expand
        x = self.swish(self.bn0(self.expand_conv(x)))

        # Depthwise
        x = self.swish(self.bn1(self.depthwise_conv(x)))

        # Squeeze and Excitation
        se = torch.mean(x, (2, 3), keepdim=True) #平均池化操作，对输入的x张量宽度和长度进行平均，得出一个
        se = self.swish(self.se_expand(self.se_reduce(se)))
        x = torch.sigmoid(se) * x

        # Project
        x = self.bn2(self.project_conv(x))

        if residual.shape == x.shape:
            x += residual

        return x

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25):
        super(FusedMBConv, self).__init__()
        mid_channels = in_channels * expand_ratio

        self.expand = nn.Sequential()
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU()
            )
        else:
            mid_channels = in_channels

        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size[0] // 2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU()
        )

        # Squeeze and Excitation
        se_mid = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, se_mid, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(se_mid, mid_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.use_res_connect = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = x * self.se(x)
        x = self.project(x)

        if self.use_res_connect:
            x = x + identity
        return x




class EfficientNetEncoder(nn.Module):
    def __init__(self,pretrained=True):
        super(EfficientNetEncoder, self).__init__()
        self._conv_stem = Conv2dStaticSamePadding(3, 40, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self._bn0 = nn.BatchNorm2d(40, eps=0.001, momentum=0.010000000000000009)

        self._blocks = nn.ModuleList([
            FusedMBConv(40, 24, expand_ratio=1, kernel_size=(3, 3), stride=1),   # Block 0 (Modified Fused-MBConv)
            FusedMBConv(24, 24, expand_ratio=1, kernel_size=(3, 3), stride=1),   # Block 1 (Modified Fused-MBConv)
            FusedMBConv(24, 32, expand_ratio=6, kernel_size=(3, 3), stride=2),   # Block 2 (Modified Fused-MBConv)
            FusedMBConv(32, 32, expand_ratio=6, kernel_size=(3, 3), stride=1, se_ratio=8/192),  # Block 3 (Modified Fused-MBConv)
            FusedMBConv(32, 32, expand_ratio=6, kernel_size=(3, 3), stride=1, se_ratio=8/192),  # Block 4 (Modified Fused-MBConv)
            FusedMBConv(32, 48, expand_ratio=6, kernel_size=(5, 5), stride=2, se_ratio = 8/192),   #  block 5
            FusedMBConv(48, 48, expand_ratio=6, kernel_size=(5, 5), stride=1, se_ratio = 12/288),   #  block 6
            FusedMBConv(48, 48, expand_ratio=6, kernel_size=(5, 5), stride=1, se_ratio = 12/288),   #  block 7
            MBConvBlock(48, 96, expand_ratio=6, kernel_size=(3, 3), stride=2, se_ratio = 12/288),   #  block 8
            MBConvBlock(96, 96, expand_ratio=6, kernel_size=(3, 3), stride=1, se_ratio = 24/576),   #  block 9
            MBConvBlock(96, 96, expand_ratio=6, kernel_size=(3, 3), stride=1, se_ratio = 24/576),   #  block 10
            MBConvBlock(96, 96, expand_ratio=6, kernel_size=(3, 3), stride=1, se_ratio = 24/576),   #  block 11
            MBConvBlock(96, 96, expand_ratio=6, kernel_size=(3, 3), stride=1, se_ratio = 24/576),   #  block 12
            MBConvBlock(96, 136, expand_ratio=6, kernel_size=(5, 5), stride=1, se_ratio = 24/576),   #  block 13
            MBConvBlock(136, 136, expand_ratio=6, kernel_size=(5, 5), stride=1, se_ratio = 34/816),   #  block 14
            MBConvBlock(136, 136, expand_ratio=6, kernel_size=(5, 5), stride=1, se_ratio = 34/816),   #  block 15
            MBConvBlock(136, 136, expand_ratio=6, kernel_size=(5, 5), stride=1, se_ratio = 34/816),   #  block 16
            MBConvBlock(136, 136, expand_ratio=6, kernel_size=(5, 5), stride=1, se_ratio = 34/816),   #  block 17
            MBConvBlock(136, 232, expand_ratio=6, kernel_size=(5, 5), stride=2, se_ratio = 34/816),   #  block 18
            MBConvBlock(232, 232, expand_ratio=6, kernel_size=(5, 5), stride=1, se_ratio = 58/1392),   #  block 19
            MBConvBlock(232, 232, expand_ratio=6, kernel_size=(5, 5), stride=1, se_ratio = 58/1392),   #  block 20
            MBConvBlock(232, 232, expand_ratio=6, kernel_size=(5, 5), stride=1, se_ratio = 58/1392),   #  block 21
            MBConvBlock(232, 232, expand_ratio=6, kernel_size=(5, 5), stride=1, se_ratio = 58/1392),   #  block 22
            MBConvBlock(232, 232, expand_ratio=6, kernel_size=(5, 5), stride=1, se_ratio = 58/1392),   #  block 23
            MBConvBlock(232, 384, expand_ratio=6, kernel_size=(3, 3), stride=1, se_ratio = 58/1392),   #  block 24
            MBConvBlock(384, 384, expand_ratio=6, kernel_size=(3, 3), stride=1, se_ratio = 96/2304),   #  block 25
            # Additional blocks would be added here...
        ])

        self._swish = nn.SiLU()  # MemoryEfficientSwish is an optimized version of SiLU

    def _load_pretrained_weights(self):
        # 使用timm库加载预训练的efficientnet-b3
        pretrained_model = timm.create_model('efficientnet_b3', pretrained=True)
        pretrained_state_dict = pretrained_model.state_dict()

        # 将预训练权重加载到自定义的encoder中
        own_state_dict = self.state_dict()
        for name, param in pretrained_state_dict.items():
            if name in own_state_dict:
                try:
                    own_state_dict[name].copy_(param)
                    print(f"Loaded pretrained weight for {name}")
                except:
                    print(f"Skipping {name} due to shape mismatch")

    def forward(self, x):
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)

        # for i, block in enumerate(self._blocks): #验证decoder中每个block输出张量的尺寸，如果需要打印则把feature那部分注释掉，该部分拿出来
        #     x = block(x)
        #     print(f'Block {i}: output shanpe: {x.shape}')

        # return x
        features = [x]

        for i, block in enumerate(self._blocks):
            x = block(x)
            if i in { 4, 7, 17, 25}:  # Capture the feature maps from certain blocks for U-Net decoder
                features.append(x)
            #print(f'Block {i}: output shape: {x.shape}')
        # features.append(x)
        return features, x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        attention_map = torch.cat([avg_out, max_out], dim=1)  # 在通道维度拼接
        attention_map = self.sigmoid(self.conv(attention_map))  # 生成注意力权重
        return x * attention_map  # 应用注意力权重

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(self):
        super(UnetDecoder, self).__init__()
        self.center = nn.Identity()  # No center block used in this architecture
        
        self.blocks = nn.ModuleList([
            DecoderBlock(520, 256),  # Decoder block 0
            DecoderBlock(304, 128),  # Decoder block 1
            DecoderBlock(160, 64),   # Decoder block 2
            DecoderBlock(104, 32),   # Decoder block 3
            DecoderBlock(32, 16),    # Decoder block 4
        ])

        # 添加空间注意力模块，仅对前两层跳跃连接添加
        self.spatial_attention_blocks = nn.ModuleList([
            SpatialAttention(),  # 对应跳跃连接第 0 层
            SpatialAttention(),  # 对应跳跃连接第 1 层
            None,                # 不对其余层添加
            None,
            None
        ])

    def forward(self, features):
        x = features[-1]
        for i in range(len(self.blocks)):
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')  # 上采样
            if i < len(features) - 1:
                skip_connection = features[-(i+2)]  # 获取跳跃连接特征

                # 在前两层添加空间注意力机制
                if self.spatial_attention_blocks[i] is not None:
                    skip_connection = self.spatial_attention_blocks[i](skip_connection)

                x = torch.cat([x, skip_connection], dim=1)  # 跳跃连接
            x = self.blocks[i](x)
        return x

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)

class UNetWithEfficientNet(nn.Module):
    def __init__(self, n_class=4, activation=None,pretrained=True):
        super(UNetWithEfficientNet, self).__init__()
        self.encoder = EfficientNetEncoder(pretrained=pretrained)  # EfficientNet编码器
        self.decoder = UnetDecoder()  # UNet解码器
        self.segmentation_head = SegmentationHead(in_channels=16, out_channels=n_class)  # 分割头

        # 添加在encoder和decoder之间的处理层
        # self.conv_head = nn.Conv2d(384, 1536, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(1536)
        # self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        # self.dropout = nn.Dropout(p=0.3)
        # self.swish = nn.SiLU()

    def forward(self, x):
        features, last_feature = self.encoder(x)  # 获取encoder输出的特征
        # 在encoder和decoder之间的处理部分
        # for i, feature in enumerate(features):
        #     print(f"Feature {i} shape: {feature.shape}") #测试模型的各特征层的准确性时用到

        # x = self.conv_head(last_feature)
        # x = self.bn1(x)
        # x = self.avg_pooling(x)
        # x = self.dropout(x)
        # x = self.swish(x)
        # 将处理后的特征图与decoder的输入进行融合
        x = self.decoder(features)
        x = self.segmentation_head(x)
        return x


if __name__ == "__main__":
    model = UNetWithEfficientNet()
    input_tensor = torch.randn(1, 3, 256, 256)  # 输入为 256x256 的 RGB 图像
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # 检查输出张量的形状
    flops = FlopCountAnalysis(model, input_tensor)


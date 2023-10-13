import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from segmentation.models.layers import ResBlock, LayerNorm


class CA(nn.Module):

    def __init__(self, k_size=3):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class MSIM(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.sca = Spatial_Channel_Aware(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size,
                                         num_heads=num_heads,
                                         channel_attn_drop=dropout_rate, spatial_attn_drop=dropout_rate)
        self.res1 = ResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.res2 = ResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))
        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.sca(self.norm(x))
        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        attn = self.res1(attn_skip)
        attn = self.res2(attn)
        x = attn_skip + self.conv(attn)
        return x


class Spatial_Channel_Aware(nn.Module):

    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)
        self.E = self.F = nn.Linear(input_size, proj_size)
        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)
        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        B, N, C = x.shape

        # four different linear layer
        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]
        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        # Spatial Reduction
        k_shared_projected = self.E(k_shared)
        v_SA_projected = self.F(v_SA)
        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        #  Channel-Aware module
        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        #  Spatial-Aware module
        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


class CCAB(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, channel_attention=False):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.channel_attention = channel_attention
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm3d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm3d(out_channels)
        if self.channel_attention:
            self.ca = CA()

    def forward(self, x):
        x_conv = self.conv1(x)
        x = self.relu(self.in1(x_conv))
        x = self.relu(self.in2(self.conv2(x)))
        if self.channel_attention:
            x = self.ca(x)
        return x_conv, x


class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownLayer, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.ccab = CCAB(in_channels, out_channels, channel_attention=True)

    def forward(self, x: torch.Tensor):
        x = self.max_pool(x)
        _, x = self.ccab(x)
        return x


class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super(UpLayer, self).__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.ccab = CCAB(in_channels, out_channels, in_channels // 2, channel_attention=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.ccab = CCAB(in_channels, out_channels, channel_attention=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        _, x = self.ccab(x)
        return x


class OutLayer(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutLayer, self).__init__(
            nn.Conv3d(in_channels, num_classes, kernel_size=1)
        )


class Shared_Layer(nn.Module):
    def __init__(self, input_size=4 * 8 * 8, dim=256, proj_size=32, depths=2, num_heads=4,
                 transformer_dropout_rate=0.15, pos_embed=True):
        super().__init__()
        self.stages = nn.ModuleList()
        self.depths = depths
        self.maxpool = nn.MaxPool3d(2, stride=2)
        for _ in range(self.depths):
            self.stages.append(MSIM(input_size=input_size, hidden_size=dim, proj_size=proj_size,
                                    num_heads=num_heads,
                                    dropout_rate=transformer_dropout_rate, pos_embed=pos_embed))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.maxpool(x)
        for i in range(self.depths):
            x = self.stages[i](x)
        return x


class CMRL_model(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 trilinear: bool = True,
                 base_c: int = 64):
        super(CMRL_model, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.trilinear = trilinear

        self.ccab1 = CCAB(in_channels, base_c, channel_attention=True)
        self.down11 = DownLayer(base_c, base_c * 2)
        self.down12 = DownLayer(base_c * 2, base_c * 4)
        self.down13 = DownLayer(base_c * 4, base_c * 8)

        self.ccab2 = CCAB(in_channels, base_c, channel_attention=True)
        self.down21 = DownLayer(base_c, base_c * 2)
        self.down22 = DownLayer(base_c * 2, base_c * 4)
        self.down23 = DownLayer(base_c * 4, base_c * 8)

        factor = 2 if self.trilinear else 1
        self.shared_layer = Shared_Layer()

        self.up11 = UpLayer(base_c * 16, base_c * 8 // factor, self.trilinear)
        self.up12 = UpLayer(base_c * 8, base_c * 4 // factor, self.trilinear)
        self.up13 = UpLayer(base_c * 4, base_c * 2 // factor, self.trilinear)
        self.up14 = UpLayer(base_c * 2, base_c, self.trilinear)

        self.up21 = UpLayer(base_c * 16, base_c * 8 // factor, self.trilinear)
        self.up22 = UpLayer(base_c * 8, base_c * 4 // factor, self.trilinear)
        self.up23 = UpLayer(base_c * 4, base_c * 2 // factor, self.trilinear)
        self.up24 = UpLayer(base_c * 2, base_c, self.trilinear)

        self.out1 = OutLayer(base_c, num_classes)
        self.out2 = OutLayer(base_c, num_classes)

    def forward(self, x: torch.Tensor, stream_id=0):

        assert stream_id == 0 or stream_id == 1, "stream id must be zero or one"
        features = []  # for image registration
        if stream_id:
            x_conv, x1 = self.ccab1(x)
            x2 = self.down11(x1)
            x3 = self.down12(x2)
            x4 = self.down13(x3)
            features.append(x_conv)
            features.append(x2)
            features.append(x3)
            x5 = self.shared_layer(x4)
            x = self.up11(x5, x4)
            x = self.up12(x, x3)
            x = self.up13(x, x2)
            x = self.up14(x, x1)
            logits = self.out1(x)
        else:
            x_conv, x1 = self.ccab2(x)
            x2 = self.down21(x1)
            x3 = self.down22(x2)
            x4 = self.down23(x3)
            features.append(x_conv)
            features.append(x2)
            features.append(x3)
            x5 = self.shared_layer(x4)
            x = self.up21(x5, x4)
            x = self.up22(x, x3)
            x = self.up23(x, x2)
            x = self.up24(x, x1)
            logits = self.out2(x)

        return features, logits
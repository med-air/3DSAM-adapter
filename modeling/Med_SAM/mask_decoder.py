import torch
import torch.nn as nn
import torch.nn.functional as F

class MLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5, scale_factor):
        # head2 = self.head2(mla_p2)
        head2 = F.interpolate(self.head2(
            mla_p2), scale_factor = scale_factor, mode='trilinear', align_corners=True)
        head3 = F.interpolate(self.head3(
            mla_p3), scale_factor = scale_factor, mode='trilinear', align_corners=True)
        head4 = F.interpolate(self.head4(
            mla_p4), scale_factor = scale_factor, mode='trilinear', align_corners=True)
        head5 = F.interpolate(self.head5(
            mla_p5), scale_factor = scale_factor, mode='trilinear', align_corners=True)
        return torch.cat([head2, head3, head4, head5], dim=1)


class VIT_MLAHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, num_classes=3,
                 norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_MLAHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels

        self.mlahead = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls = nn.Sequential(nn.Conv3d(4 * mlahead_channels + 1, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, num_classes, 3, padding=1, bias=False))

    def forward(self, inputs, scale_factor=None):
        if scale_factor == None:
            scale_factor = self.img_size / inputs[0].shape[-1]
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3], scale_factor = scale_factor)
        x = torch.cat([x, inputs[-1]], dim=1)
        x = self.cls(x)
        return x


class VIT_MLAHead_h(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, num_classes=2,
                 norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_MLAHead_h, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels

        self.mlahead = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls = nn.Sequential(nn.Conv3d(4 * mlahead_channels + 1, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, num_classes, 3, padding=1, bias=False))

    def forward(self, inputs, scale_factor1, scale_factor2):
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3], scale_factor = scale_factor1)
        x = torch.cat([x, inputs[-1]], dim=1)
        x = self.cls(x)
        x = F.interpolate(x, scale_factor = scale_factor2, mode='trilinear', align_corners=True)
        return x
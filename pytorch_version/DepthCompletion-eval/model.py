import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpBlock, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.leakyreluA(self.convA(torch.cat([up_x, concat_with], dim=1)))))


class Decoder(nn.Module):
    def __init__(self, num_features=1024, feature_base=256,  decoder_width=0.5):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.inconv = nn.Conv2d(num_features+64, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpBlock(skip_input=features//1  + 64 + feature_base, output_features=features // 2)
        self.up2 = UpBlock(skip_input=features//2  + 32 + feature_base//2, output_features=features // 4)
        self.up3 = UpBlock(skip_input=features//4  + 16 +  feature_base//4, output_features=features // 8)
        self.up4 = UpBlock(skip_input=features//8  +  8 +  feature_base//4, output_features=features // 16)
        self.up5 = UpBlock(skip_input=features//16 +  4 +   3, output_features=features // 32)

        self.outconv = nn.Conv2d(features//32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features_color, features_depth):

        c_in, c_block0, c_block1, c_block2, c_block3, c_block4 = \
            features_color[0], features_color[3], features_color[4], \
            features_color[6], features_color[8], features_color[11]

        d_in, d_block0, d_block1, d_block2, d_block3, d_block4 = \
            features_depth[0], features_depth[1], features_depth[2], \
            features_depth[3], features_depth[4], features_depth[5]

        x_d0 = self.inconv(torch.cat([c_block4, d_block4], dim=1))      #     1024  * 512 -> 1/32

        x_d1 = self.up1(x_d0, torch.cat([c_block3, d_block3], dim=1))  # (512+256) * 256 -> 1/16

        x_d2 = self.up2(x_d1, torch.cat([c_block2, d_block2], dim=1))  # (128+128) * 128 -> 1/8

        x_d3 = self.up3(x_d2, torch.cat([c_block1, d_block1], dim=1))  # (64 + 64) *  64 -> 1/4

        x_d4 = self.up4(x_d3, torch.cat([c_block0, d_block0], dim=1))  # (32 + 64) *  32 -> 1/2
        x_d5 = self.up5(x_d4, torch.cat([c_in, d_in], dim=1))  # (32 + 64) *  32 -> 1/2
        return self.outconv(x_d5)


class Encoder(nn.Module):
    def __init__(self, densenet='121'):
        super(Encoder, self).__init__()
        import torchvision.models as models
        if densenet == '161':
            self.original_model = models.densenet161(pretrained=True)
            print('Use Pretrain Densenet161 Model.')
        else:
            self.original_model = models.densenet121(pretrained=True, memory_efficient=False)
            print('Use Pretrain Densenet121 Model.')

        for k, v in self.original_model.named_parameters():
            v.requires_grad = False  # 固定参数

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features


class DownBlock(nn.Sequential):
    def __init__(self, input, output_features):
        super(DownBlock, self).__init__()
        self.convA = nn.Conv2d(input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)
        self.pool = nn.AvgPool2d(kernel_size=2, padding=0, ceil_mode=False)

    def forward(self, x):
        return self.leakyreluB(self.convB(self.leakyreluA(self.convA(self.pool(x)))))


class InBlock(nn.Sequential):
    def __init__(self, input, output_features):
        super(InBlock, self).__init__()
        self.convA = nn.Conv2d(input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leakyreluB(self.convB(self.leakyreluA(self.convA(x))))


class Encoder_Depth(nn.Module):
    def __init__(self):
        super(Encoder_Depth, self).__init__()

        self.In = InBlock(1, 4)
        self.d0 = DownBlock(4, 8)
        self.d1 = DownBlock(8, 16)
        self.d2 = DownBlock(16, 32)
        self.d3 = DownBlock(32, 64)
        self.d4 = DownBlock(64, 64)

    def forward(self, features):
        x_in = self.In(features)
        x_d0 = self.d0(x_in)
        x_d1 = self.d1(x_d0)
        x_d2 = self.d2(x_d1)
        x_d3 = self.d3(x_d2)
        x_d4 = self.d4(x_d3)
        return [x_in, x_d0, x_d1, x_d2, x_d3, x_d4]


class Model(nn.Module):
    def __init__(self, pretrain_model='121'):
        super(Model, self).__init__()
        if pretrain_model == '121':
            self.encoder = Encoder(densenet='121')
            self.decoder = Decoder(num_features=1024, feature_base=256)
        else:
            self.encoder = Encoder(densenet='161')
            self.decoder = Decoder(num_features=2208, feature_base=384)

        self.encoder_depth = Encoder_Depth()

    def forward(self, x):
        color = x[:, :3, :, :]
        depth = x[:, 3:4, :, :]
        return self.decoder(self.encoder(color), self.encoder_depth(depth))


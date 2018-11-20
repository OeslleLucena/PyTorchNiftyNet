import torch.nn as nn
import torch
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes, base_filter=8):
        super(UNet3D, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.base_n_filter = base_filter

        # Encoder side
        #1
        self.conv_bn_relu_block_1 = self.conv_bn_relu_block(self.in_channel,self.base_n_filter)
        self.conv_bn_relu_block_1_2 = self.conv_bn_relu_block(self.base_n_filter, self.base_n_filter*2)
        self.max_pool_1 = nn.MaxPool3d(2)

        #2
        self.conv_bn_relu_block_2 = self.conv_bn_relu_block(self.base_n_filter*2,self.base_n_filter*2)
        self.conv_bn_relu_block_2_2 = self.conv_bn_relu_block(self.base_n_filter*2, self.base_n_filter*4)
        self.max_pool_2 = nn.MaxPool3d(2)

        # Latent space
        self.conv_bn_relu_block_3 = self.conv_bn_relu_block(self.base_n_filter*4,self.base_n_filter*4)
        self.conv_bn_relu_block_3_2 = self.conv_bn_relu_block(self.base_n_filter*4, self.base_n_filter*8)

        # Decoder side
        #1
        self.conv_1 = self.conv(self.base_n_filter*4+self.base_n_filter*8, self.base_n_filter*4)
        self.conv_bn_relu_block_4 = self.conv_bn_relu_block(self.base_n_filter*4, self.base_n_filter*4)
        self.conv_bn_relu_block_4_1 = self.conv_bn_relu_block(self.base_n_filter*4, self.base_n_filter*4)

        #2
        self.conv_2 = self.conv(self.base_n_filter*2 + self.base_n_filter*4, self.base_n_filter*2)
        self.conv_bn_relu_block_5 = self.conv_bn_relu_block(self.base_n_filter*2, self.base_n_filter*2)
        self.conv_bn_relu_block_5_1 = self.conv_bn_relu_block(self.base_n_filter*2, self.base_n_filter*2)

        # output
        self.ouput = self.conv(self.base_n_filter*2,n_classes,kernel_size=1)


    def conv_bn_relu_block(self, in_channels, out_channels, kernel_size=3,
                           stride=1,  bias=False):
        padding = (kernel_size - 1) // 2
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())

        return layer

    def conv(self, in_channels, out_channels, kernel_size=3,
                           stride=1, bias=False):
        padding = (kernel_size - 1) // 2
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias))

        return layer


    def forward(self, x):

        # Encoder 1
        out = self.conv_bn_relu_block_1(x)
        conv_bn_relu_block_1_2 = self.conv_bn_relu_block_1_2(out)
        out = self.max_pool_1(conv_bn_relu_block_1_2)

        # Encoder 2
        out = self.conv_bn_relu_block_2(out)
        conv_bn_relu_block_2_2 = self.conv_bn_relu_block_2_2(out)
        out = self.max_pool_2(conv_bn_relu_block_2_2)

        # Latent space
        out = self.conv_bn_relu_block_3(out)
        out = self.conv_bn_relu_block_3_2(out)

        # Decoder 1
        out = F.upsample(out, scale_factor=2, mode='trilinear')
        out = torch.cat((out, conv_bn_relu_block_2_2),1)
        out = self.conv_1(out)
        out = self.conv_bn_relu_block_4(out)
        out = self.conv_bn_relu_block_4_1(out)

        # Decoder 2
        out = F.upsample(out, scale_factor=2, mode='trilinear')
        out = torch.cat((out, conv_bn_relu_block_1_2),1)
        out = self.conv_2(out)
        out = self.conv_bn_relu_block_5(out)
        out = self.conv_bn_relu_block_5_1(out)

        # Output
        out = self.ouput(out)

        return F.sigmoid(out)

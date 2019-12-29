import torch.nn as nn
import math

def _conv2d_bn(in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    bn = nn.BatchNorm2d(num_features=out_channels)
    return nn.Sequential(conv, bn)


def _conv2d_bn_relu(in_channels, out_channels, kernel_size, stride, padding):
    conv2d_bn = _conv2d_bn(in_channels, out_channels, kernel_size, stride, padding)
    relu = nn.ReLU(inplace=True)
    layers = list(conv2d_bn.children())
    layers.append(relu)
    return nn.Sequential(*layers)


class _BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num, skip_conn, downscale=False):
        super(_BasicBlock, self).__init__()
        self.skip_conn = skip_conn
        self.conv_num = conv_num
        self.down_sampler = None
        stride = 1
        if downscale:
            self.down_sampler = _conv2d_bn(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
            stride = 2
        
        if self.conv_num == 2 :
            self.conv_bn_relu1 = _conv2d_bn_relu(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.conv_bn2 = _conv2d_bn(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.conv_num == 3 :
            self.conv_bn_relu1 = _conv2d_bn_relu(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.conv_bn_relu2 = _conv2d_bn_relu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)   
            self.conv_bn3 = _conv2d_bn(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.conv_num == 1 :
            self.conv_bn1 = _conv2d_bn(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        if self.down_sampler:
            input = self.down_sampler(x)
        if self.conv_num == 2 :
            residual = self.conv_bn_relu1(x)
            residual = self.conv_bn2(residual)
        if self.conv_num == 3 :
            residual = self.conv_bn_relu1(x)
            residual = self.conv_bn_relu2(residual)
            residual = self.conv_bn3(residual)
        if self.conv_num == 1 :
            residual = self.conv_bn1(x)
      
        if self.skip_conn :
            out = self.relu_out(residual)
        else :
            out = self.relu_out(input + residual)
        return out

class _ResNet(nn.Module):
#     def __init__(self, num_layer_stack):
    def __init__(self, layers_num, out_f, conv_num, skip_conn):
        
        # define resnet_n
        # num_layer_stack = int((layers_num - 2) / 6)
        
        super(_ResNet, self).__init__()
        self.conv1 = _conv2d_bn_relu(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.layer1 = self.__make_layers(layers_num[0], in_channels=16, out_channels=16, conv_num=conv_num, skip_conn=skip_conn, downscale=False)
        self.layer2 = self.__make_layers(layers_num[1], in_channels=16, out_channels=32, conv_num=conv_num, skip_conn=skip_conn, downscale=True)
        self.layer3 = self.__make_layers(layers_num[2], in_channels=32, out_channels=64, conv_num=conv_num, skip_conn=skip_conn, downscale=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(in_features=64, out_features=out_f)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def __make_layers(self, num_layer_stack, in_channels, out_channels, conv_num, skip_conn, downscale):
        layers = []
        layers.append(_BasicBlock(in_channels=in_channels, out_channels=out_channels, conv_num=conv_num, skip_conn=skip_conn, downscale=downscale))
        for i in range(num_layer_stack - 1):
            layers.append(_BasicBlock(in_channels=out_channels, out_channels=out_channels, conv_num=conv_num, skip_conn=skip_conn, downscale=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y

def resnet_n(layers_num, out_f, conv_num = 2, skip_conn = False):
        return _ResNet(layers_num, out_f, conv_num, skip_conn)
    
# def resnet10_20():
#     return _ResNet(num_layer_stack=3)


# def resnet10_32():
#     return _ResNet(num_layer_stack=5)


# def resnet10_56():
#     return _ResNet(num_layer_stack=9)

# def resnet10_110():
#     return _ResNet(num_layer_stack=18)

# 6x + 2 = layers, x = num_layer_stack to define model
# x = (layers - 2) / 6
# layers = 20, 56, 110
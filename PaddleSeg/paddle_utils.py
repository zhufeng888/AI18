import paddle
import paddle.nn as nn

class conv_bn_relu(nn.Layer):
    def __init__(self,conv, in_channels,out_channels,kernel_size,is_bn, bn, is_relu, stride=1,padding=0,
                 dilation=1,groups=1,padding_mode='zeros', weight_attr=None,bias_attr=None, data_format='NCHW'):
        super(conv_bn_relu, self).__init__()
        self.is_bn = is_bn
        self.is_relu = is_relu
      
        self.conv = conv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                   padding_mode, weight_attr, bias_attr, data_format)
        if is_bn:
            self.bn = bn(num_features=out_channels)

        if is_relu:
            self.activation = nn.ReLU()


    def forward(self, input, features2=None):
        input = self.conv(input)
        if self.is_bn:
            input = self.bn(input)
        if self.is_relu:
            input = self.activation(input)
        return input

class conv2d_bn_relu(conv_bn_relu):
    def __init__(self, in_channels,out_channels,kernel_size,is_bn=True,bn=nn.BatchNorm2D, is_relu=True, stride=1,
                 padding=0,dilation=1,groups=1,padding_mode='zeros', weight_attr=None,bias_attr=None, data_format='NCHW'):

        super().__init__(conv=nn.Conv2D, in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                         is_bn=is_bn, bn=bn,is_relu=is_relu,
                         stride=stride,padding=padding,dilation=dilation,groups=groups,padding_mode=padding_mode,
                         weight_attr=weight_attr,bias_attr=bias_attr, data_format=data_format)


class conv1d_bn_relu(conv_bn_relu):
    def __init__(self, in_channels, out_channels, kernel_size, is_bn=True, bn=nn.BatchNorm1D, is_relu=True, stride=1,
                 padding=0,dilation=1, groups=1, padding_mode='zeros', weight_attr=None,bias_attr=None, data_format='NCHW'):
        super().__init__(conv=nn.Conv1D, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         is_bn=is_bn, bn=bn, is_relu=is_relu,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode,
                         weight_attr=weight_attr,bias_attr=bias_attr, data_format=data_format)

class down_conv2d_bn_relu(nn.Layer):
    def __init__(self, in_channels,out_channels,kernel_size,is_bn=True,bn=nn.BatchNorm2D, is_relu=True, stride=1,
                 padding=0,dilation=1,groups=1,padding_mode='zeros', weight_attr=None,bias_attr=None, data_format='NCHW',
                 down='max', down_size=2, down_stride=None):
        super(down_conv2d_bn_relu, self).__init__()
        
        if down == 'max':
            self.down = nn.MaxPool2D(kernel_size=down_size, stride=down_stride)

        self.conv2d_bn_relu = conv2d_bn_relu(in_channels,out_channels,kernel_size,is_bn=is_bn,bn=bn, is_relu=is_relu, stride=stride,
                 padding=padding,dilation=dilation,groups=groups,padding_mode=padding_mode,weight_attr=weight_attr,bias_attr=bias_attr, data_format=data_format)

    def forward(self, features, features2=None):
        features = self.down(features)

        features = self.conv2d_bn_relu(features)
        return features


class up_conv2d_bn_relu(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, is_bn=True, bn=nn.BatchNorm2D, is_relu=True, stride=1,
                 padding=0, dilation=1, groups=1, padding_mode='zeros', weight_attr=None,bias_attr=None, data_format='NCHW',
                 up='bilinear', up_size=2,up_stride=1):
        super(up_conv2d_bn_relu, self).__init__()

        self.groups = groups
        self.up = up

        self.conv2d_bn_relu = conv2d_bn_relu(in_channels, out_channels, kernel_size, is_bn=is_bn, bn=bn,
                                             is_relu=is_relu, stride=stride,
                                             padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode,
                                             weight_attr=weight_attr,bias_attr=bias_attr, data_format=data_format)

    def forward(self, features, features2):


        if self.up=='nearst':
            up_type = nn.UpsamplingNearest2D(features2.shape[-2:])
        if self.up=='bilinear':
            up_type = nn.UpsamplingBilinear2D(features2.shape[-2:])

        features = up_type(features)


        features_list = []
        if self.groups > 1:
            assert features.shape[1] % self.groups == 0 and features2.shape[1] % self.groups == 0

            interval1 = int(features.shape[1]/self.groups)
            features_split_list = paddle.split(features, interval1,1)
            interval2 = int(features2.shape[1] / self.groups)
            features2_split_list = paddle.split(features2, interval2, 1)
            for i in range(self.groups):
                features_list.append(features_split_list[i])
                features_list.append(features2_split_list[i])
        else:
            features_list.append(features)
            features_list.append(features2)

        features = paddle.concat(features_list, 1)

        features = self.conv2d_bn_relu(features)
        return features


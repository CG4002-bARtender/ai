# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class SmallResNet(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(SmallResNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #SmallResNet::input_0(SmallResNet::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SmallResNet::SmallResNet/Sequential[stem]/Conv2d[0]/ret.3(SmallResNet::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #SmallResNet::SmallResNet/Sequential[stem]/ReLU[2]/1885(SmallResNet::nndct_relu_2)
        self.module_3 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[0]/Conv2d[conv1]/ret.7(SmallResNet::nndct_conv2d_3)
        self.module_4 = py_nndct.nn.ReLU(inplace=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[0]/ReLU[relu]/1913(SmallResNet::nndct_relu_4)
        self.module_5 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[0]/Conv2d[conv2]/ret.11(SmallResNet::nndct_conv2d_5)
        self.module_6 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[0]/Sequential[skip]/Conv2d[0]/ret.15(SmallResNet::nndct_conv2d_6)
        self.module_7 = py_nndct.nn.Add() #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[0]/ret.19(SmallResNet::nndct_elemwise_add_7)
        self.module_8 = py_nndct.nn.ReLU(inplace=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[0]/ReLU[relu]/1971(SmallResNet::nndct_relu_8)
        self.module_9 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[0]/MaxPool2d[pool]/1986(SmallResNet::nndct_maxpool_9)
        self.module_10 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[1]/Conv2d[conv1]/ret.21(SmallResNet::nndct_conv2d_10)
        self.module_11 = py_nndct.nn.ReLU(inplace=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[1]/ReLU[relu]/2015(SmallResNet::nndct_relu_11)
        self.module_12 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[1]/Conv2d[conv2]/ret.25(SmallResNet::nndct_conv2d_12)
        self.module_13 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[1]/Sequential[skip]/Conv2d[0]/ret.29(SmallResNet::nndct_conv2d_13)
        self.module_14 = py_nndct.nn.Add() #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[1]/ret.33(SmallResNet::nndct_elemwise_add_14)
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[1]/ReLU[relu]/2073(SmallResNet::nndct_relu_15)
        self.module_16 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[1]/MaxPool2d[pool]/2088(SmallResNet::nndct_maxpool_16)
        self.module_17 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[2]/Conv2d[conv1]/ret.35(SmallResNet::nndct_conv2d_17)
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[2]/ReLU[relu]/2117(SmallResNet::nndct_relu_18)
        self.module_19 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[2]/Conv2d[conv2]/ret.39(SmallResNet::nndct_conv2d_19)
        self.module_20 = py_nndct.nn.Add() #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[2]/ret.43(SmallResNet::nndct_elemwise_add_20)
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #SmallResNet::SmallResNet/Sequential[blocks]/ResBlock[2]/ReLU[relu]/2149(SmallResNet::nndct_relu_21)
        self.module_22 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #SmallResNet::SmallResNet/AdaptiveAvgPool2d[gap]/2199(SmallResNet::nndct_adaptive_avg_pool2d_22)
        self.module_23 = py_nndct.nn.Module('nndct_flatten') #SmallResNet::SmallResNet/ret.45(SmallResNet::nndct_flatten_23)
        self.module_24 = py_nndct.nn.Linear(in_features=128, out_features=10, bias=True) #SmallResNet::SmallResNet/Linear[fc]/ret(SmallResNet::nndct_dense_24)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_3 = self.module_3(output_module_0)
        output_module_3 = self.module_4(output_module_3)
        output_module_3 = self.module_5(output_module_3)
        output_module_6 = self.module_6(output_module_0)
        output_module_3 = self.module_7(input=output_module_3, other=output_module_6, alpha=1)
        output_module_3 = self.module_8(output_module_3)
        output_module_3 = self.module_9(output_module_3)
        output_module_10 = self.module_10(output_module_3)
        output_module_10 = self.module_11(output_module_10)
        output_module_10 = self.module_12(output_module_10)
        output_module_13 = self.module_13(output_module_3)
        output_module_10 = self.module_14(input=output_module_10, other=output_module_13, alpha=1)
        output_module_10 = self.module_15(output_module_10)
        output_module_10 = self.module_16(output_module_10)
        output_module_17 = self.module_17(output_module_10)
        output_module_17 = self.module_18(output_module_17)
        output_module_17 = self.module_19(output_module_17)
        output_module_17 = self.module_20(input=output_module_17, other=output_module_10, alpha=1)
        output_module_17 = self.module_21(output_module_17)
        output_module_17 = self.module_22(output_module_17)
        output_module_17 = self.module_23(input=output_module_17, start_dim=1, end_dim=-1)
        output_module_17 = self.module_24(output_module_17)
        return output_module_17

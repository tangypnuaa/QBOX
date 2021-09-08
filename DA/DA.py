import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from .LabelResizeLayer import ImageLabelResizeLayer
from .LabelResizeLayer import InstanceLabelResizeLayer


class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        # ctx.alpha = alpha
        ctx.alpha = 0.1
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output


def grad_reverse(x):
    return GRLayer.apply(x)

# class ImageDA(nn.Module):

class GridDA(nn.Module):

    def __init__(self, dim):
        super(GridDA, self).__init__()
        self.dim = dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=False)
        self.Conv2 = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=False)
        self.reLu = nn.ReLU(inplace=False)
        self.LabelResizeLayer = ImageLabelResizeLayer()

    def forward(self, x, domain_label):
        x = grad_reverse(x)
        x = self.reLu(self.Conv1(x))
        x = self.Conv2(x)
        label = self.LabelResizeLayer(x, domain_label)
        return x, label

class _MultiScaleDA(nn.Module):
    def __init__(self, dim, dim1, dim2):
        super(_MultiScaleDA, self).__init__()
        self.dim = dim  # feat layer          256*H*W for vgg16
        self.dim1, self.dim2 = dim1, dim2
        self.dim_trans1 = nn.Conv2d(self.dim1, self.dim, kernel_size=1, bias=False)
        self.dim_trans2 = nn.Conv2d(self.dim2, self.dim, kernel_size=1, bias=False)

        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=False)
        self.Conv2 = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=False)
        self.reLu = nn.ReLU(inplace=False)
        self.LabelResizeLayer = ImageLabelResizeLayer()

    def forward(self, x, need_backprop):
        # x is list(Tensor) with channels 256, 512, 1024
        x1, x2, x3 = x

        x3 = grad_reverse(x3)
        x1 = grad_reverse(x1)
        x2 = grad_reverse(x2)

        x2 = self.dim_trans1(x2)
        x3 = self.dim_trans2(x3)

        x1 = self.reLu(self.Conv1(x1))
        x2 = self.reLu(self.Conv1(x2))
        x3 = self.reLu(self.Conv1(x3))
        x1 = self.Conv2(x1)
        x2 = self.Conv2(x2)
        x3 = self.Conv2(x3)
        label1 = self.LabelResizeLayer(x1, need_backprop)
        label2 = self.LabelResizeLayer(x2, need_backprop)
        label3 = self.LabelResizeLayer(x3, need_backprop)
        return (x1, x2, x3), (label1, label2, label3)


class _InstanceDA(nn.Module):
    def __init__(self):
        super(_InstanceDA, self).__init__()
        self.dc_ip1 = nn.Linear(2048, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer = nn.Linear(1024, 1)
        self.LabelResizeLayer = InstanceLabelResizeLayer()

    def forward(self, x, need_backprop):
        x = grad_reverse(x)
        x = self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x = self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x = F.sigmoid(self.clssifer(x))
        label = self.LabelResizeLayer(x, need_backprop)
        return x, label

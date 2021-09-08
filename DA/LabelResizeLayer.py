import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torch.autograd import Function
import cv2


class ImageLabelResizeLayer(nn.Module):
    """
    Resize label to be the same size with the samples
    """

    def __init__(self):
        super(ImageLabelResizeLayer, self).__init__()

    def forward(self, x, domain_label):
        lab = torch.zeros((x.shape[0], x.shape[2], x.shape[3])).to(domain_label.device)
        lab += torch.max(domain_label)
        return lab.long()


class InstanceLabelResizeLayer(nn.Module):

    def __init__(self):
        super(InstanceLabelResizeLayer, self).__init__()
        self.minibatch = 256

    def forward(self, x, need_backprop):
        feats = x.data.cpu().numpy()
        lbs = need_backprop.data.cpu().numpy()

        resized_lbs = np.ones((feats.shape[0], 1), dtype=np.float32)
        for i in range(lbs.shape[0]):
            resized_lbs[i * self.minibatch:(i + 1) * self.minibatch] = lbs[i]

        y = torch.from_numpy(resized_lbs).cuda()

        return y

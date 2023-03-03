import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


if __name__ == "__main__":
    net = GradLayer()
    img_dir = "welds/"
    out_dir = "sobel_out/"
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(img_dir, filename)
            img = cv2.imread(filepath).astype(np.float32)/255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            img = net(img)
            img = (img[0, :, :, :].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            outpath = os.path.join(out_dir, filename)
            cv2.imwrite(outpath, img)

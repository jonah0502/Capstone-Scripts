import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np

# Define a custom PyTorch module for the Sobel filter
class SobelFilter(nn.Module):
    
    def __init__(self):
        super(SobelFilter, self).__init__()
        
        # Define the Sobel kernels for horizontal and vertical edges
        kernel_h = torch.tensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.tensor([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Define the kernels as parameters of the module
        self.weight_h = nn.Parameter(kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(kernel_v, requires_grad=False)

    def get_gray(self,x):
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)
    
    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)
        # Apply the Sobel kernels to the input tensor using PyTorch's conv2d function
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x_v = F.conv2d(x, self.weight_v, padding=1)
        
        # Calculate the magnitude of the Sobel gradient
        x = torch.sqrt(torch.pow(x_h, 2) + torch.pow(x_v, 2))
        
        return x

# Create an instance of the SobelFilter module
sobel_filter = SobelFilter()

# Set the directory paths for input and output images
input_dir = "welds"
output_dir = "sobel_v2_out"

# Iterate over all image files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image from file using OpenCV
        img = cv2.imread(os.path.join(input_dir, filename))
        
        # Convert the image from OpenCV's default BGR format to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert the image to a PyTorch tensor with float32 data type and normalize the pixel values to the range [0, 1]
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        
        # Apply the Sobel filter to the image using the SobelFilter module
        img = sobel_filter(img)
        
        # Convert the output tensor back to a numpy array and scale the pixel values back to the range [0, 255] with uint8 data type
        img = (img[0, :, :, :].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        
        # Save the output image to file in the output directory
        cv2.imwrite(os.path.join(output_dir, filename), img)
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid


class DataLoader(nn.Module):
    pass

# Hyperparameters

h_img = 240
w_img = 240
patch_size = 4





class Patchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x => bs, h, w, c
        bs, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2)

        x = self.unfold(x)
        # Reshaping into the shape we want
        a = x.view(bs, c, self.p, self.p, -1).permute(0, 4, 2, 3, 1)
        print(f'a shape {a.shape}')
        return a


patch = Patchify()

convlayer = nn.Conv2d(in_channels=3, out_channels=96, stride=patch_size, kernel_size=patch_size)


image = cv2.imread(
    "/Users/sisirdhakal/Documents/PersonalProjects/NanoGpt/460631282_1058085612351815_4727211779822854898_n.jpg"
)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype("float32") / 255.0  # Normalize to [0, 1]
image = torch.from_numpy(image)
image = image.unsqueeze(0)

print(f'image shape {image.shape}')

p = patch(image)

# print(image.shape)
print(f'patch {p.shape}')
# print(p)

image = image.permute(0, 3, 1, 2)  # Now it's [1, 3, 960, 960]
test = convlayer(image)

print(test.shape)



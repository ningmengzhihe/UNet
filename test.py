import os
import torch
from utils import keep_image_size_open
from torchvision.utils import save_image
from net import *
from data import *


net = UNet().cuda()

weights = 'params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')

# 在这里输入图片地址，例如：E:/Workspace/Unet/test_image/000033.jpg
_input = input('please input image path:')
img = keep_image_size_open(_input)
img_data = transform(img).cuda()
print(img_data.shape)

img_data = torch.unsqueeze(img_data, dim=0)  # 增加一个batch的维度
out = net(img_data)
save_image(out, 'result/result.jpg')
print(out)



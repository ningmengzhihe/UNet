from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import *
from net import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
# PASCAL VOC2012 dataset and set batch_size=1 on DataLoader
data_path = 'E:/Workspace/UNet/VOCdevkit/VOC2012'
# for test and set batch_size=8 on DataLoader
# data_path = 'E:/Workspace/UNet/testdata'
save_path = 'train_image'


if __name__ == '__main__':
    # Note:如果CUDA out of memory试着调小batch_size
    # 如果shuffle设置成True，那么0_0.png和1_0.png不是同一张图片，可能是shuffle设置成了随机的原因
    # 如果shuffle设置成False，那么0_0.png和1_0.png是同一张图片了，可能是限制了每个batch中图片的顺序固定不变
    data_loader = DataLoader(MyDataset(data_path), batch_size=8, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    n_epoch = 2
    for epoch in range(0, n_epoch):
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i%5 == 0:
                # 每隔5次打印一次loss
                print(f'{epoch}-{i}-train_loss ===>> {train_loss.item()}')

            if i%50 == 0:
                # 每隔50个batch保存一次权重
                torch.save(net.state_dict(), weight_path)

                # 每隔50个batch保存batch的第一张图片

                _image = image[0]
                _segment_image = segment_image[0]
                _out_image = out_image[0]

                img = torch.stack([_image, _segment_image, _out_image], dim=0)
                save_image(img, f'{save_path}/{epoch}_{i}.png')


import os

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
transform = transforms.Compose([ # 一般用Compose把多个步骤整合到一起
    transforms.ToTensor()
])
# 将dataset类中__getitem__()方法内读入的PIL或CV的图像数据转换为torch.FloatTensor

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # Note:原始数据中的JPEGImages比SegmentationClass图片多很多张，这个代码中只用了SegmentationClass图片去检索JPEGImages原图
        segment_name = self.name[index]  ## xx.png
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))  # 原图地址
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)  # 原图等比缩放
        return transform(image), transform(segment_image)


if __name__ == '__main__':
    data = MyDataset('E:/Workspace/UNet/testdata') # Python在windows下的标准路径是斜杠’ / ’ ,但是仍然可以识别 反斜杠’ \ ’，如果路径中有'\Uxxx'或者'\txxx'的时候会识别不出来，建议使用规范的路径写法'/'
    print(data[0][0].shape)  # 打印第1张图的原图形状
    print(data[0][1].shape)  # 打印第1张图的分割图形状







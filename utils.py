from PIL import Image
# Python图像库PIL(Python Image Library)是python的第三方图像处理库，但是由于其强大的功能与众多的使用人数，几乎已经被认为是python官方图像处理库了。


def keep_image_size_open(path, size=(256,256)):
    '''
    # 等比缩放
    :return:
    '''
    img = Image.open(path)
    temp = max(img.size) # 图片最长边
    mask = Image.new('RGB', (temp, temp), (0,0,0)) # 用图片最长边做一个矩形，黑色
    mask.paste(img, (0,0)) # 粘贴到左上角
    mask = mask.resize(size)
    return mask
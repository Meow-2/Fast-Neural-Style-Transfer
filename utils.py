from torchvision import transforms
import torch
import numpy as np
import av

# Mean and standard deviation used for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def extract_frames(video_path):
    """ Extracts frames from video """
    frames = []
    video = av.open(video_path)
    for frame in video.decode(0):
        yield frame.to_image()


def gram_matrix(y):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    # 用于计算两个矩阵的批量乘法, 除以元素个数
    gram = features.bmm(features_t) / (c * h * w)
    # (b, c, c)
    return gram 


def train_transform(image_size):
    """ Transforms for training images """
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform


# image_size 是一个整数, 可以将图像的长边调整到该值, 图像长宽比不变
# 返回的transform是一个转换函数, 可以将图片 resize, ToTensor, Normalize(mean,std)
def style_transform(image_size=None):
    """ Transforms for style image """
    # transforms.Resize() 返回的是一个用来 resize 的函数
    resize = [transforms.Resize(image_size)] if image_size else []
    # transforms.Compose() 可以将多个图像转换函数合在一起构成一个转换序列
    transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transform


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors


def deprocess(image_tensor):
    """ Denormalizes and rescales image tensor """
    image_tensor = denormalize(image_tensor)[0]
    image_tensor *= 255
    image_np = torch.clamp(image_tensor, 0, 255).cpu().numpy().astype(np.uint8)
    image_np = image_np.transpose(1, 2, 0)
    return image_np

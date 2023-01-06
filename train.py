import argparse
import os
import sys
import random
from PIL import Image
import numpy as np
import torch
import glob
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from models import TransformerNet, VGG16
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for Fast-Neural-Style")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to training dataset")
    parser.add_argument("--style_image", type=str, default="style-images/mosaic.jpg", help="path to style image")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=112, help="Size of training images")
    parser.add_argument("--style_size", type=int, help="Size of style image")
    parser.add_argument("--lambda_content", type=float, default=1e5, help="Weight for content loss")
    parser.add_argument("--lambda_style", type=float, default=1e10, help="Weight for style loss")
    parser.add_argument("--lambda_triplet", type=float, default=1e10, help="Weight for style loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint_model", type=str, help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Batches between saving model")
    parser.add_argument("--sample_interval", type=int, default=1000, help="Batches between saving image samples")
    args = parser.parse_args()

    # 获取风格图的名称
    style_name = args.style_image.split("/")[-1].split(".")[0]
    os.makedirs(f"images/outputs/{style_name}-training", exist_ok=True)
    os.makedirs(f"checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataloader for the training data
    train_dataset = datasets.ImageFolder(args.dataset_path, train_transform(args.image_size))
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    # Defines networks
    transformer = TransformerNet().to(device)
    vgg = VGG16(requires_grad=False).to(device)

    # Load checkpoint model if specified
    if args.checkpoint_model:
        transformer.load_state_dict(torch.load(args.checkpoint_model))

    # Define optimizer and loss
    optimizer = Adam(transformer.parameters(), args.lr)
    l2_loss = torch.nn.MSELoss().to(device)

    # Load style image
    # 风格图被 resize, transforms.ToTensor(), transforms.Normalize(mean, std)
    # transforms.ToTensor() 会转换通道并对像素进行归一化即除以255
    style = style_transform(args.style_size)(Image.open(args.style_image))
    # tensor.repeat 将 tensor 在第 0 维重复 batch_size 次, 一个batch的 style_image
    # 目的是为了维度上进行匹配
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # Extract style features
    # vgg 输出的是一个 namedtuple(relu1_2,relu2_2,relu3_3,relu4_3)
    # 可以通过 features.relu1_2 ... 调用
    features_style = vgg(style)
    # 对于每个 style features 求 gram_style
    # gram_style = [gram(relu1_2), gram(relu2_2), gram(relu3_3), gram(relu4_3)]
    # gram.shape (bz, c, c)
    gram_style = [gram_matrix(y) for y in features_style]

    # 取随机采样 8 张图片用于评估模型
    # Sample 8 images for visual evaluation of the model
    image_samples = []
    # glob 会匹配路径下所有符合的图片, 并返回一个列表
    # random.sample 表示从列表里随机取 8 张图
    for path in random.sample(glob.glob(f"{args.dataset_path}/*/*.png"), 8):
        image_samples += [style_transform(args.image_size)(Image.open(path))]
    # image_samples 里存的是转化为tensor的8张随机采样的图片
    # torch.stack 用于将一组 tensor 按照指定维度堆叠起来
    # 也是模拟一个batch
    image_samples = torch.stack(image_samples)

    def save_sample(batches_done):
        """ Evaluates the model and saves image samples """
        # 模型转为评估模式
        transformer.eval()
        # 测试一下这些图片
        with torch.no_grad():
            output = transformer(image_samples.to(device))
        # 输出8张图的变化过程
        image_grid = denormalize(torch.cat((image_samples.cpu(), output.cpu()), 2))
        save_image(image_grid, f"images/outputs/{style_name}-training/{batches_done}.jpg", nrow=4)
        transformer.train()

    for epoch in range(args.epochs):
        epoch_metrics = {"content": [], "style": [], "total": []}
        for batch_i, (images, _) in enumerate(dataloader):
            # 梯度清零
            optimizer.zero_grad()

            # 图片放到 tensor 里
            images_original = images.to(device)
            # 输出转换后的图片
            images_transformed = transformer(images_original)

            # Extract features
            features_original = vgg(images_original)
            features_transformed = vgg(images_transformed)

            # Compute content loss as MSE between features
            # relu2_2 作为 内容损失
            content_loss = args.lambda_content * l2_loss(features_transformed.relu2_2, features_original.relu2_2)

            # Compute style loss as MSE between gram matrices
            style_loss = 0
            # zip 用于将两个同样长度的容器拼接
            # features_transformed 是含有四个 tensor 的元组, gram_style 是含有四个 tensor 的列表
            for ft_y, gm_s in zip(features_transformed, gram_style):
                # ft_y (b,c,h,w) gm_s(b,c,c)
                # gm_y (b,c,c)
                gm_y = gram_matrix(ft_y)
                # gm_s 放缩到 batch 的大小, 因为 dataloader 最后的一个 batch 可能没有 batch_size 的大小
                style_loss += l2_loss(gm_y, gm_s[: images.size(0), :, :])
            style_loss *= args.lambda_style

            # 计算 triplet loss
            # ==================================================================
            # triplet_loss = 
            # triplet_loss = triplet_loss*args.lambda_triplet
            # total_loss = content_loss + style_loss + triplet_loss
            # ==================================================================
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            epoch_metrics["content"] += [content_loss.item()]
            epoch_metrics["style"] += [style_loss.item()]
            epoch_metrics["total"] += [total_loss.item()]

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Content: %.2f (%.2f) Style: %.2f (%.2f) Total: %.2f (%.2f)]"
                % (
                    epoch + 1,
                    args.epochs,
                    batch_i,
                    len(train_dataset),
                    content_loss.item(),
                    np.mean(epoch_metrics["content"]),
                    style_loss.item(),
                    np.mean(epoch_metrics["style"]),
                    total_loss.item(),
                    np.mean(epoch_metrics["total"]),
                )
            )

            # 已经完成的 batch 数量
            batches_done = epoch * len(dataloader) + batch_i + 1
            # 每隔 1000 个 batch 测试一次效果
            if batches_done % args.sample_interval == 0:
                save_sample(batches_done)

            # 每隔 1000 个 batch 保存一次模型
            if args.checkpoint_interval > 0 and batches_done % args.checkpoint_interval == 0:
                style_name = os.path.basename(args.style_image).split(".")[0]
                torch.save(transformer.state_dict(), f"checkpoints/{style_name}_{batches_done}.pth")

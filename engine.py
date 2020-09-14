import argparse
import os
import random
import time

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from models import TransformerNet, Vgg16

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(weights=args.vgg16, requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        count = 0
        if args.noise_count:
            noiseimg_n = np.zeros((3, args.image_size, args.image_size), dtype=np.float32)
            # Preparing noise image.
            for n_c in range(args.noise_count):
                x_n = random.randrange(args.image_size)
                y_n = random.randrange(args.image_size)
                noiseimg_n[0][x_n][y_n] += random.randrange(-args.noise, args.noise)
                noiseimg_n[1][x_n][y_n] += random.randrange(-args.noise, args.noise)
                noiseimg_n[2][x_n][y_n] += random.randrange(-args.noise, args.noise)
                noiseimg = torch.from_numpy(noiseimg_n)
                noiseimg = noiseimg.to(device) 
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
         
            x = x.to(device)
            if args.noise_count:
                # Adding the noise image to the source image.
                noisy_x = x + noiseimg
                noisy_y = transformer(noisy_x)
                noisy_y = utils.normalize_batch(noisy_y)

            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            L_feat = args.lambda_feat * mse_loss(features_y.relu2_2, features_x.relu2_2)

            L_style = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                L_style += mse_loss(gm_y, gm_s[:n_batch, :, :])
            L_style *= args.lambda_style

            L_tv = (
                torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) 
                + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
            )
            
            L_tv *= args.lambda_tv

            if args.noise_count:
                L_pop = args.lambda_noise * F.mse_loss(y, noisy_y)
                L = L_feat + L_style + L_tv + L_pop
                print('Epoch {},{}/{}. Total loss: {}. Loss distribution: feat {}, style {}, tv {}, pop {}'
                            .format(e, batch_id, len(train_loader), L.data,
                                    L_feat.data/L.data, L_style.data/L.data,
                                    L_tv.data/L.data, L_pop.data/L.data))
            else:
                L = L_feat + L_style + L_tv
                print('Epoch {},{}/{}. Total loss: {}. Loss distribution: feat {}, style {}, tv {}'
                            .format(e, batch_id, len(train_loader), L.data,
                                    L_feat.data/L.data, L_style.data/L.data,
                                    L_tv.data/L.data)) 
            L = L_style*1e10 + L_feat*1e5
            L.backward()
            optimizer.step()

    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)

def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    style_model = TransformerNet()
    state_dict = torch.load(args.model)
    style_model.load_state_dict(state_dict)
    style_model.to(device)

    img_list = os.listdir(args.content_dir)
    img_list.sort()
    for img in tqdm(img_list):
        img_path = args.content_dir + img
        content_org = utils.load_image(img_path, scale=args.content_scale)
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_org)
        content_image = content_image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = style_model(content_image).cpu()
        output = output[0]
        output = output.clone().clamp(0, 255).numpy()
        output = output.transpose(1, 2, 0).astype("uint8")
        output = Image.fromarray(output)

        if args.keep_colors:
            med = utils.original_colors(content_org, output)

        output.save(args.output_dir + img)
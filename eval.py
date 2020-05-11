import os
import re

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

import utils
from transformer_net import TransformerNet

def val(args):
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

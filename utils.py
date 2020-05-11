import os
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

def check_paths(args, train=True):
    if train:
        try:
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
                os.makedirs(args.checkpoint_model_dir)
        except OSError as e:
            print(e)
            sys.exit(1)
    else:
        try:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
        except OSError as e:
            print(e)
            sys.exit(1)

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')
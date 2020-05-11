from __future__ import print_function
import numpy as np
import argparse
from PIL import Image, ImageFilter
import time
import torch
import os

parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
#parser.add_argument('input')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', default='models/style.model', type=str)
parser.add_argument('--out', '-o', default='out.jpg', type=str)
parser.add_argument('--median_filter', default=0, type=int)
parser.add_argument('--padding', default=50, type=int)
parser.add_argument('--keep_colors', action='store_true')
parser.set_defaults(keep_colors=True)
args = parser.parse_args()

# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')

from transformer_net import TransformerNet
device = torch.device("cuda")
with torch.no_grad():
    style_model = TransformerNet()
    state_dict = torch.load(args.model)
    style_model.load_state_dict(state_dict)
    style_model.to(device)

model = FastStyleNet()
model.load_state_dict(torch.load('./ckpt/mosaic_epoch_2_Thu_Feb_13_21:02:56_2020.model'))
model.cuda()
out_dir = './out_vid2/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)  
start = time.time()
all_frames = os.listdir('./frames/')
all_frames.sort()
for index in all_frames:  
    args.input = './frames/' + index
    original = Image.open(args.input).convert('RGB')
    image = np.asarray(original, dtype=np.float32).transpose(2, 0, 1)
    image = image.reshape((1,) + image.shape)
    if args.padding > 0:
        image = np.pad(image, [[0, 0], [0, 0], [args.padding, args.padding], [args.padding, args.padding]], 'symmetric')
    image = np.asarray(image)
    x = torch.from_numpy(image).cuda()
    y = model(x)
    result = y.data.cpu()
    if args.padding > 0:
        result = result[:, :, args.padding:-args.padding, args.padding:-args.padding]
    result = np.uint8(result[0]).transpose(1, 2, 0)
    med = Image.fromarray(result)
    if args.median_filter > 0:
        med = med.filter(ImageFilter.MedianFilter(args.median_filter))
    if args.keep_colors:
        med = original_colors(original, med)
    print(time.time() - start, 'sec')
    med.save(out_dir + 'frame' + str(val) + '.png')

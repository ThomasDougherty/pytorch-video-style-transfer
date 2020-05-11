#!/bin/bash

#python main.py eval --content-dir ./frames/ --model ./ckpt/epoch_2_Tue_Mar__3_22:07:35_2020.model --output-dir ./style_frames/style_4/ --cuda 1
#ffmpeg -f image2 -framerate 30 -i %*.png video.avi

import os

ckpt_dir = './ckpt/'
models = [
'epoch_2_Mon_Apr__6_00:07:17_2020.model'
]
for model_name in models:
    print('Model: ' + model_name)
    filename, ext = os.path.splitext(model_name)
    out_dir = './style_frames/' + filename + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    extract_frames = 'python main.py eval --content-dir ./frames/ --model ./ckpt/' + model_name + ' --output-dir ./style_frames/' + filename + '/ --cuda 1'
    make_video = 'ffmpeg -f image2 -framerate 30 -i ' + out_dir + '%*.png ' + out_dir + 'video.avi'
    os.system(extract_frames)
    os.system(make_video)


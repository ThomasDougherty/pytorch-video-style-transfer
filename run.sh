#!/bin/bash
python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1
python main.py eval --content-dir ./frames/ --model ./ckpt/epoch_2_Sat_Feb_22_02:17:41_2020.model --output-dir ./style_frames/style_2/ --cuda 1


ffmpeg -f image2 -framerate 30 -i %*.png video.avi
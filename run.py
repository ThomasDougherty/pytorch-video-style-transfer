#!/bin/bash

#python main.py eval --content-dir ./frames/ --model ./ckpt/epoch_2_Tue_Mar__3_22:07:35_2020.model --output-dir ./style_frames/style_4/ --cuda 1
#ffmpeg -f image2 -framerate 30 -i %*.png video.avi

import os

'''cmd_list = [
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 500 --noise 30 --noise-count 1000 --lambda_tv 10e-6',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1000 --noise 30 --noise-count 1000 --lambda_tv 10e-6',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1500 --noise 30 --noise-count 1000 --lambda_tv 10e-6',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 500 --noise 60 --noise-count 1000 --lambda_tv 10e-6',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1000 --noise 60 --noise-count 1000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 200 --noise 30 --noise-count 2000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 500 --noise 30 --noise-count 1500 --lambda_tv 10e-6',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1000 --noise 30 --noise-count 1500 --lambda_tv 10e-6',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1500 --noise 30 --noise-count 2000 --lambda_tv 10e-6',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1500 --noise 60 --noise-count 2000 --lambda_tv 10e-6',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 500 --noise 30 --noise-count 2000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1000 --noise 30 --noise-count 2000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 2000 --noise 30 --noise-count 2000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 500 --noise 60 --noise-count 2000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1000 --noise 60 --noise-count 2000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 200 --noise 30 --noise-count 2000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 500 --noise 30 --noise-count 2000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1000 --noise 30 --noise-count 2000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1500 --noise 30 --noise-count 2000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/mosaic.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1500 --noise 60 --noise-count 2000 --lambda_tv 10e-5',
]'''

'''cmd_list = [
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/rain-princess-cropped.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1500 --noise 60 --noise-count 4000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/rain-princess-cropped.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 3000 --noise 60 --noise-count 2000 --lambda_tv 10e-5',
****'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/rain-princess-cropped.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1500 --noise 100 --noise-count 2000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/rain-princess-cropped.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 3000 --noise 100 --noise-count 4000 --lambda_tv 10e-5',
]'''

# cmd_list = [
# 'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/candy.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1500 --noise 60 --noise-count 4000 --lambda_tv 10e-5',
# 'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/candy.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 3000 --noise 60 --noise-count 2000 --lambda_tv 10e-5',
# 'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/candy.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 1500 --noise 100 --noise-count 2000 --lambda_tv 10e-5',
# 'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/candy.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 3000 --noise 100 --noise-count 4000 --lambda_tv 10e-5', ** CURRENT BEST MODEL
# ]

cmd_list = [
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/candy.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 3000 --noise 100 --noise-count 8000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/candy.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 3000 --noise 200 --noise-count 4000 --lambda_tv 10e-5',
*** WINNER'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/candy.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 3000 --noise 200 --noise-count 8000 --lambda_tv 10e-5',
'python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/candy.jpg  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 5000 --noise 200 --noise-count 8000 --lambda_tv 10e-5',
]

for cmd in cmd_list:
    print(cmd)
    os.system(cmd)

python main.py train --dataset ../../../data/coco/train2014/ --style-image ./style-images/elves.png  --save-model-dir ./ckpt/ --cuda 1 --lambda-feat 10 --lambda-style 1 --lambda-noise 3000 --noise 200 --noise-count 8000 --lambda_tv 10e-5
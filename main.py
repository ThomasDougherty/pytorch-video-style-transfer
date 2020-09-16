#!/usr/bin/env python3
import argparse
import torch

from engine import train, stylize
import utils

def get_args_parser():
    main_arg_parser = argparse.ArgumentParser(description="parser for video_style_transfer")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                    help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=3,
                                    help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                    help="path to training dataset, the path should point to a folder "
                                        "containing another folder with all the training images")
    train_arg_parser.add_argument("--vgg16", type=str, default=None,
                                    help="path to vgg16 weights, if left None then torchvision will try install")
    train_arg_parser.add_argument("--style-image", type=str, required=True,
                                    help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                    help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--image-size", type=int, default=512,
                                    help="size of training images, default is 512 X 512")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                    help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                    help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                    help="random seed for training")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                    help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                    help="number of images after which the "
                                         "training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                    help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                    help="number of batches after which a "
                                         "checkpoint of the trained model will "
                                         "be created")
    train_arg_parser.add_argument("--lambda_tv", default=1e-8, type=float,
                                    help=("weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.")
                                 )
    train_arg_parser.add_argument("--lambda-feat", default=1, type=float)
    train_arg_parser.add_argument("--lambda-style", default=1.0, type=float)
    train_arg_parser.add_argument("--lambda-noise", default=50.0, type=float,
                                    help="Training weight of the popping induced by noise")
    train_arg_parser.add_argument("--noise", default=30, type=int,
                        help="range of noise for popping reduction")
    train_arg_parser.add_argument("--noise-count", default=250, type=int,
                        help="number of pixels to modify with noise")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-dir", type=str, required=True,
                                    help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                    help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-dir", type=str, required=True,
                                    help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                    help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                    help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--keep-colors", action="store_true",
                                    help="transfer color to stylized output")
    return main_arg_parser.parse_args()

def main():
    args = get_args_parser()
    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    
    if args.subcommand == "train":
        print('Starting train...')
        utils.check_paths(args)
        train(args)
    else:
        print('Starting stylization...')
        utils.check_paths(args, train=False)
        stylize(args)

if __name__ == '__main__':
    main()
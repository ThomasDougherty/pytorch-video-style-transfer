from __future__ import print_function, division

import torch

from eval import val
from options import parse_args
from train import train
import utils

def main():
    args = parse_args()
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
        val(args)

if __name__ == '__main__':
    main()
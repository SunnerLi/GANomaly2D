from lib.utils import INFO, showParameters
import argparse
import torch
import os

def parse_args(phase = 'train'):
    """
        Parse the argument for training procedure
        ------------------------------------------------------
                        [Argument explain]
        
        --train         : The folder path of training image (only normal)
        --demo          : The folder path of inference image (normal + abnormal)
        --resume        : The path of pre-trained model
        --det           : The path of destination model you want to store into
        --H             : The height of image
                          Default is 240
        --W             : The width of image
                          Default is 320
        --r             : The ratio of channel you want to reduce
                          Default is 1
        --batch_size    : The batch size in single batch
                          Default is 2
        --n_iter        : Total iteration
                          Default is 1 (30000 is recommand)
        --record_iter   : The period to record the render image and model parameters
                          Default is 1 (200 is recommand)

        ------------------------------------------------------
        Arg:    phase   (Str)   - The symbol of program (train or demo)
        Ret:    The argparse object    
    """
    parser = argparse.ArgumentParser()
    if phase == 'train':
        parser.add_argument('--train'           , type = str, required = True)
    if phase == 'demo':
        parser.add_argument('--demo'            , type = str, required = True)
    parser.add_argument('--resume'          , type = str, default = 'result.pkl')
    parser.add_argument('--det'             , type = str, default = 'result.pkl')
    parser.add_argument('--H'               , type = int, default = 240)
    parser.add_argument('--W'               , type = int, default = 320)
    parser.add_argument('--r'               , type = int, default = 1)
    parser.add_argument('--batch_size'      , type = int, default = 1)
    parser.add_argument('--n_iter'          , type = int, default = 1)
    parser.add_argument('--record_iter'     , type = int, default = 1)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    showParameters(vars(args))
    return args

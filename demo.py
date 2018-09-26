from parse.parse import parse_args 
from lib.visualize import visualizeAnomalyImage, visualizeEncoderDecoder
from lib.model import GANomaly2D

import torchvision_sunner.transforms as sunnerTransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms

from tqdm import tqdm
import argparse
import torch
import cv2
import os

"""
    This script defines the demo procedure of GANomaly2D

    Author: SunnerLi
"""

def demo(args):
    """
        This function define the demo process
        
        Arg:    args    (napmespace) - The arguments
    """
    # Create the data loader
    loader = sunnerData.DataLoader(
        dataset = sunnerData.ImageDataset(
            root = [[args.demo]], 
            transform = transforms.Compose([
                sunnerTransforms.Resize(output_size = (args.H, args.W)),
                sunnerTransforms.ToTensor(),
                sunnerTransforms.ToFloat(),
                sunnerTransforms.Transpose(),
                sunnerTransforms.Normalize(),
            ])
        ), batch_size = args.batch_size, shuffle = True, num_workers = 2
    )

    # Create the model
    model = GANomaly2D(r = args.r, device = args.device)
    model.IO(args.resume, direction = 'load')

    # Demo!
    bar = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for (img,) in bar:
            z, z_ = model.forward(img)
            img, img_ = model.getImg()            
            visualizeAnomalyImage(img, img_, z, z_) 
        
if __name__ == '__main__':
    args = parse_args(phase = 'demo')
    demo(args)
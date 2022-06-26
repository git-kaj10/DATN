import os
import cv2
import random
import numpy as np
import torch 
import argparse 
from shutil import copyfile
from src.config import Config
from src.edge_connect import EdgeConnect 


def load_config(mode=None):
    r"""loads model config

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')

    # test mode
    parser.add_argument('--input', type=str,default='./Images/Inputs', help='path to the input images directory or an input image')
    parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
    parser.add_argument('--output', type=str,default='./Images/Outputs', help='path to the output directory')
    parser.add_argument('--remove', nargs= '*' ,type=int, help='objects to remove')
    parser.add_argument('--cpu', type=str,default='yes', help='machine to run segmentation model on')
    parser.add_argument(
        '-f',
        '--file',
        help='Path for input file. First line should contain number of lines to search in'
    )
    args = parser.parse_args()
    
    #if path for checkpoint not given
    if args.path is None:
        args.path='./checkpoints'
    config_path = os.path.join(args.path, 'config.yml')
    
       # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

   
    # test mode
    config.MODE = 2
    config.MODEL = args.model if args.model is not None else 3
    config.OBJECTS = args.remove if args.remove is not None else [3,15]
    config.SEG_DEVICE = 'cpu' if args.cpu is not None else 'cuda'
    config.INPUT_SIZE = 256
    if args.input is not None:
        config.TEST_FLIST = args.input
    
    if args.edge is not None:
        config.TEST_EDGE_FLIST = args.edge
    if args.output is not None:
        config.RESULTS = args.output
    else: 
        if not os.path.exists('./results_images'):
            os.makedirs('./results_images')
        config.RESULTS = './results_images'
    
    
    print('------------ Options -------------')
    print("INPUT_SIZE:", config.INPUT_SIZE)
    print("TEST_FLIST:", config.TEST_FLIST)
    print("RESULTS:", config.RESULTS)
    print("MODE:", config.MODE)
    print("MODEL:", config.MODEL)
    print("OBJECTS:", config.OBJECTS)
    print("SEG_DEVICE:", config.SEG_DEVICE)
    print("INPUT_SIZE:", config.INPUT_SIZE)
    
    print('-------------- End ----------------')
    
    
    return config
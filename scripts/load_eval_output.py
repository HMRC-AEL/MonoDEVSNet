import os

import torch
import argparse
from tqdm import tqdm

from utils import depth_visualize

parser = argparse.ArgumentParser(description="MonoDEVSNet test options")
parser.add_argument("--file_path",
                    type=str,
                    help="path to the MonoDEVSNet's precompute depth maps",
                    default='')
opts = parser.parse_args()


# rgbs: num_images x height x width x channels
# depth: num_images x 1 x height x width
def load_eval_output(pth_file_path):
    data = torch.load(pth_file_path)
    rgbs = data['rgbs']
    pred_depths = data['pred_depths']
    for i in tqdm(range(len(rgbs))):
        rgb = rgbs[i]
        pred_depth = pred_depths[i]
        depth_visualize(rgb, pred_depth, title_=str(i))


if __name__ == '__main__':
    try:
        load_eval_output(opts.file_path)
    except Exception as e:
        print(e)

# Author: Akhil Gurram
# Build on top of the monodepth2
# (Automatically pulled from git repo, monodepth2 source code is not included in this repository)
# This is the training script of the MonoDEVSNet framework.
# MonoDEVSNet: Monocular Depth Estimation through Virtual-world Supervision and Real-world SfM Self-Supervision
# https://arxiv.org/abs/2103.12209

# MIT License
#
# Copyright (c) 2021 Huawei Technologies Duesseldorf GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import json
import os
import sys
import time
from datetime import date
from os.path import expanduser

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from matplotlib import cm, pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import networks
from monodepth2.datasets import KITTIRAWDataset, KITTIDepthDataset
from utils import get_n_params, MonoDEVSOptions, convert_list2dict, readlines
from utils.utils_local import Dict2Struct

home = expanduser("~")
week_num = date.today().isocalendar()[1]
turbo = cm.get_cmap('turbo')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class MonoDEVSTestOptions(MonoDEVSOptions):
    def __init__(self, *args, **kwargs):
        super(MonoDEVSTestOptions, self).__init__(*args, **kwargs)

        # PATHS
        self.parser.add_argument("--image_folder_path",
                                 type=str,
                                 help="path to real dataset, for dataset=\"any\" option provide the images folder",
                                 default="")
        self.parser.add_argument("--do_kb_crop",
                                 help="decide to crop the region defined by KITTI benchmark or not",
                                 action="store_true")
        self.options = []

    def parse(self):
        self.options = self.parser.parse_args()
        if self.options.load_weights_folder is None:
            self.options.load_weights_folder = os.path.join('models', self.options.config)  # The Best model path

        changed_names = convert_list2dict(self.options.models_fcn_name)
        default_class = {"encoder": "HRNet", "depth_decoder": "DepthDecoder",
                         "pose_encoder": "ResnetEncoder", "pose_decoder": "PoseDecoder",
                         "domain_classifier": "DomainClassifier", "dis_depth": "DepthDiscriminator",
                         "gan_s_decoder": "ImageDecoder", "gan_t_decoder": "ImageDecoder",
                         "dis_s": "ImageDiscriminator", "dis_t": "ImageDiscriminator"}
        for k, v in changed_names.items():
            default_class[k] = v

        self.options.models_fcn_name = default_class.copy()
        return self.options


# Computation of error metrics between predicted and ground truth depths
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


class Evaluation(object):
    def __init__(self, opt, model_name=None):

        # Load experiments options/parameters
        self.opt = opt
        self.model_name = model_name
        try:
            with open(os.path.join(self.opt.load_weights_folder, 'opt.json')) as file:
                self.opt_from_load_path = Dict2Struct(**json.load(file))
            self.opt.height = self.opt_from_load_path.height
            self.opt.width = self.opt_from_load_path.width
            self.opt.min_depth = self.opt_from_load_path.min_depth
            self.opt.max_depth = self.opt_from_load_path.max_depth
            self.opt.num_layers = self.opt_from_load_path.num_layers
        except Exception as e:
            print(e)

        self.opt.trainer_name = 'trainer.py'

        torch.autograd.set_detect_anomaly(False)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        assert self.opt.load_weights_folder is not None, "load weight folder path shouldn\'t be empty, provide path"

        self.models = {}

        # Set cuda index
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.opt.cuda_idx)
        self.device = torch.device("cuda:" + str(self.opt.cuda_idx) if torch.cuda.is_available() else "cpu")

        # Load network architecture
        ''' Setting Models, initialization and optimization '''
        # Encoder for (depth, segmentation, pose)
        self.models["encoder"] = self.network_selection('encoder')

        # Depth decoder
        self.models["depth_decoder"] = self.network_selection('depth_decoder')

        # Loading pretrained model
        if self.opt.load_weights_folder is not None:
            try:
                self.load_pretrained_models()
                print('Loaded MonoDEVSNet trained model')
            except Exception as e:
                print(e)
                print('models not found, start downloading!')
                sys.exit(0)

        if not os.path.exists(self.opt.log_dir):
            os.makedirs(self.opt.log_dir)

        if self.model_name is None:
            self.model_name = self.opt.models_fcn_name['encoder'] + '_' + str(self.opt.num_layers)
        print('Exp name: {}'.format(model_name))

        print("\nFiles are saved to:\n  ", self.opt.log_dir)
        print("Running scripts on :  ", self.device)
        print("Number of parameters for each model")
        for model_name, model in self.models.items():
            print('{:^15}: {:>5.2f} M'.format(model_name, get_n_params(model) / 1000000))

        # Images path list
        img_ext = '.png' if self.opt.png else '.jpg'
        self.im_path_list = []
        if self.opt.dataset == 'kitti':
            real_eigen = readlines(os.path.join(os.path.dirname(__file__), "splits", "eigen", "test_files.txt"))
            dataset = KITTIRAWDataset(data_path=self.opt.real_data_path, filenames=real_eigen,
                                      height=self.opt.height, width=self.opt.width,
                                      frame_idxs=[0], num_scales=4, is_train=False,
                                      img_ext=img_ext)
            self.dataloader = DataLoader(dataset, self.opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                         pin_memory=True, drop_last=False)
            print('Total number of images in {} dataset: {}'.format(self.opt.dataset, dataset.__len__()))
        if self.opt.dataset == 'kitti_depth':
            real_eigen = readlines(os.path.join(os.path.dirname(__file__), "splits", "eigen", "test_files.txt"))
            dataset = KITTIDepthDataset(data_path=self.opt.real_data_path, filenames=real_eigen,
                                        height=self.opt.height, width=self.opt.width,
                                        frame_idxs=[0], num_scales=4, is_train=False,
                                        img_ext=img_ext)
            self.dataloader = DataLoader(dataset, self.opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                         pin_memory=True, drop_last=False)
            print('Total number of images in {} dataset: {}'.format(self.opt.dataset, dataset.__len__()))
        elif self.opt.dataset == 'any':
            for base_path, base_folder, file_paths in os.walk(self.opt.image_folder_path):
                for file_path in sorted(file_paths):
                    if file_path.endswith('.png') or file_path.endswith('.jpg'):
                        self.im_path_list.append(os.path.join(base_path, file_path))
            print('Total number of images in {} dataset: {}'.format(self.opt.dataset, len(self.im_path_list)))
        else:
            raise RuntimeError('Choose dataset to test MonoDEVSNet model')

        self.rgbs, self.pred_depths, self.gt_depths = [], [], []
        self.resize = transforms.Resize((self.opt.height, self.opt.width), interpolation=Image.ANTIALIAS)
        if self.opt.dataset == 'kitti':
            # Eigen split - LIDAR data
            gt_path = os.path.join(os.path.dirname(__file__), "splits", "eigen", "gt_depths.npz")
            self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    def invdepth_to_depth(self, inv_depth):
        return 1 / self.opt.max_depth + (1 / self.opt.min_depth - 1 / self.opt.max_depth) * inv_depth

    def eval_any(self):
        errors_absolute, time_taken, time_for_each_frame, flops_for_each_frame, total_time = \
            [], time.time(), [], [], time.time()
        with torch.no_grad():
            for iter_l, im_path in tqdm(enumerate(self.im_path_list)):
                try:
                    input_color, input_color_np = self.load_rgb_image(im_path)
                except Exception as e:
                    print(e)
                    print('failed image path: {}'.format(im_path))
                height_o, width_o, channels_o = input_color_np.shape

                time_taken = time.time()
                # Modify this accordingly - network arch
                features, _ = self.models["encoder"](input_color)
                output = self.models["depth_decoder"](features)
                time_taken -= time.time()
                time_for_each_frame.append(np.abs(time_taken))

                # Convert disparity into depth maps
                pred_disp = self.invdepth_to_depth(output[("disp", 0)])
                pred_disp = pred_disp[0, 0].cpu().numpy()
                pred_depth_raw = 3. / pred_disp.copy()

                # save resized rgb,and raw pred depth
                self.rgbs.append(input_color_np)
                self.pred_depths.append(pred_depth_raw)
                pred_depth_t = torch.tensor(pred_depth_raw).unsqueeze(0).unsqueeze(0)
                pred_depth_t[pred_depth_t < self.opt.min_depth] = self.opt.min_depth
                pred_depth_t[pred_depth_t > self.opt.max_depth] = self.opt.max_depth

                # Save information
                folder_name = os.path.dirname(im_path).split('/')[-1]
                depth_save_path = im_path.replace(folder_name + '/',
                                                  folder_name + '_' + 'depth_MonoDEVSNet' + '/'). \
                    replace('.jpg', '.png')
                pred_depth_o = Image.fromarray(np.array(F.interpolate(pred_depth_t,
                                                                      (height_o, width_o)).squeeze() * 256,
                                                        dtype=np.uint16))
                pred_depth_color = Image.fromarray(np.array(turbo(F.interpolate(pred_depth_t,
                                                                                (height_o, width_o),
                                                                                mode='bilinear').squeeze() /
                                                                  self.opt.max_depth)[:, :, :3] * 255, dtype=np.uint8))

                if not os.path.exists(os.path.dirname(depth_save_path)):
                    os.makedirs(os.path.dirname(depth_save_path))
                pred_depth_o.save(depth_save_path)
                pred_depth_color.save(depth_save_path.replace('.png', '_color.png'))

        print('time taken for network model {}-{}: {}'.format(self.opt.models_fcn_name['encoder'], self.opt.num_layers,
                                                              1 / np.mean(time_for_each_frame)))
        return None, None

    def eval(self):
        if self.opt.dataset == 'any':
            return self.eval_any()
        else:
            return self.eval_local()

    def eval_local(self):
        errors_absolute, errors_relative, ratios, time_taken, time_for_each_frame, total_time = \
            [], [], [], time.time(), [], time.time()
        data_iter, total_invalid_images, iter_l = iter(self.dataloader), 0, -1
        with torch.no_grad():
            for __ in tqdm(range(self.dataloader.__len__())):
                try:
                    data = data_iter.__next__()
                    iter_l += 1
                except Exception as _:
                    total_invalid_images += 1
                    continue

                # Related to depth, segmentation, edges
                input_color = data[("color", 0, 0)].to(self.device)

                time_taken = time.time()
                features, _ = self.models["encoder"](input_color)
                output = self.models["depth_decoder"](features)
                time_taken -= time.time()
                time_for_each_frame.append(np.abs(time_taken))

                # Convert disparity into depth maps
                pred_disp = self.invdepth_to_depth(output[("disp", 0)])
                pred_disp = pred_disp[0, 0].cpu().numpy()
                pred_depth_raw = 3. / pred_disp.copy()

                if 'kitti' in self.opt.dataset:
                    if self.opt.dataset == 'kitti_depth':
                        self.gt_depths.append(data['depth_gt'][0, 0].cpu().numpy())

                gt_depth = self.gt_depths[iter_l]
                gt_height, gt_width = gt_depth.shape
                pred_disp = cv2.resize(pred_disp, (gt_width, gt_height), cv2.INTER_NEAREST)
                pred_depth = self.opt.syn_scaling_factor / pred_disp.copy()

                if self.opt.do_kb_crop:
                    crop_height, crop_width = 352, 1216
                    if gt_height == 192 or gt_width == 640:
                        crop_height, crop_width = int(crop_height / 2), int(crop_width / 2)

                    # AdaBins setting
                    top_margin, left_margin = gt_height - crop_height, int((gt_width - crop_width) / 2)
                    pred_depth = pred_depth[top_margin:top_margin + crop_height, left_margin:left_margin + crop_width]
                    gt_depth = gt_depth[top_margin:top_margin + crop_height, left_margin:left_margin + crop_width]
                else:
                    top_margin, left_margin = 0, 0
                    crop_height, crop_width = gt_depth.shape

                # Eigen crop
                mask = np.logical_and(gt_depth > self.opt.min_depth, gt_depth < self.opt.max_depth)
                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

                gt_depth[gt_depth < self.opt.min_depth] = self.opt.min_depth
                gt_depth[gt_depth > self.opt.max_depth] = self.opt.max_depth
                pred_depth[pred_depth < self.opt.min_depth] = self.opt.min_depth
                pred_depth[pred_depth > self.opt.max_depth] = self.opt.max_depth

                errors_absolute.append(compute_errors(gt_depth[mask], pred_depth[mask]))

                # save resized rgb,and raw pred depth
                self.rgbs.append(input_color.squeeze().cpu().permute(1, 2, 0).numpy().copy())
                self.pred_depths.append(pred_depth_raw)

        if 'kitti' in self.opt.dataset:
            errors_absolute = np.array(errors_absolute).mean(0)

            print('\n  \n  for {} meters - MonoDEVSNet Absolute depth estimation results '
                  'fps: {}, total time taken: {:4.4} (in mins) invalid images: {} kb_crop: {} dataset: {}'.
                  format(self.opt.max_depth, 1 / np.mean(time_for_each_frame), (time.time() - total_time) / 60,
                         total_invalid_images, str(self.opt.do_kb_crop), self.opt.dataset))
            print("  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.4f}  " * 7).format(*errors_absolute.tolist()) + "\\\\")
            torch.save({'rgbs': self.rgbs, 'pred_depths': self.pred_depths},
                       os.path.join(self.opt.log_dir, 'monoDEVSNet_kitti_eigen_test_split_' + self.model_name +
                                    '_' + self.opt.version + '.pth'))
        else:
            torch.save({'rgbs': self.rgbs, 'pred_depths': self.pred_depths},
                       os.path.join(self.opt.log_dir, 'any_data.pth'))

        return errors_absolute, errors_relative

    def load_rgb_image(self, file_path):
        if self.opt.dataset == 'kitti' or self.opt.dataset == 'any':
            im_pil = Image.open(file_path).convert('RGB')
            return torch.tensor(np.array(self.resize(im_pil), dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(
                self.device), np.array(im_pil)

    def load_pretrained_models(self):
        # Paths to the models
        encoder_path = os.path.join(self.opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(self.opt.load_weights_folder, "depth_decoder.pth")

        # Load model weights
        encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
        self.models["encoder"].load_state_dict({k: v for k, v in encoder_dict.items()
                                                if k in self.models["encoder"].state_dict()})
        self.models["depth_decoder"].load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))

        # Move network weights from cpu to gpu device
        self.models["encoder"].to(self.device).eval()
        self.models["depth_decoder"].to(self.device).eval()

    # model_key is CaSe SenSiTivE
    def network_selection(self, model_key):
        if model_key == 'encoder':
            # Multiple network architectures
            if 'HRNet' in self.opt.models_fcn_name[model_key]:
                with open(os.path.join('configs', 'hrnet_w' + str(self.opt.num_layers) + '_vk2.yaml'), 'r') as cfg:
                    config = yaml.safe_load(cfg)
                return networks.HRNetPyramidEncoder(config).to(self.device)
            elif 'DenseNet' in self.opt.models_fcn_name[model_key]:
                return networks.DensenetPyramidEncoder(densnet_version=self.opt.num_layers).to(self.device)
            elif 'ResNet' in self.opt.models_fcn_name[model_key]:
                return networks.ResnetEncoder(self.opt.num_layers,
                                              self.opt.weights_init == "pretrained").to(self.device)
            else:
                raise RuntimeError('Choose a depth encoder within available scope')

        elif model_key == 'depth_decoder':
            return networks.DepthDecoder(self.models["encoder"].num_ch_enc).to(self.device)

        else:
            raise RuntimeError("Don\'t forget to mention what you want!")


if __name__ == "__main__":
    # Evaluation on selected models
    opts = MonoDEVSTestOptions(base_path=os.path.join(os.path.dirname(os.path.abspath(__file__))))
    opts = opts.parse()
    eval_main = Evaluation(opt=opts)
    eval_main.eval()

    stop_here = 1

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
from torchvision import transforms
from tqdm import tqdm

import networks
from utils import readlines
from utils import MonoDEVSOptions, convert_list2dict

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
        self.options = []

    def parse(self):
        self.options = self.parser.parse_args()
        if self.options.load_weights_folder is None:
            self.options.load_weights_folder = os.path.join('models', 'hrnet_w48_vk2')  # The Best model path

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
        self.opt.trainer_name = 'trainer.py'

        torch.autograd.set_detect_anomaly(False)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        assert self.opt.load_weights_folder is not None, "load weight folder path shouldn\'t be empty, provide path"
        if self.opt.dataset == 'kitti':
            assert self.opt.image_folder_path != '', "kitti base folder path shouldn\'t be empty, provide path"

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

        # Images path list
        self.im_path_list = []
        if self.opt.dataset == 'kitti':
            self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
            filenames = readlines(os.path.join(self.opt.base_path, "splits", "eigen", "test_files.txt"))
            for line in filenames:
                folder, frame_index, side = line.split(' ')
                f_str = "{:010d}{}".format(int(frame_index), '.png')
                image_path = os.path.join(self.opt.image_folder_path, folder,
                                          "image_0{}/data".format(self.side_map[side]), f_str)
                self.im_path_list.append(image_path)
        elif self.opt.dataset == 'any':
            for base_path, base_folder, file_paths in os.walk(self.opt.image_folder_path):
                for file_path in sorted(file_paths):
                    if file_path.endswith('.png') or file_path.endswith('.jpg'):
                        self.im_path_list.append(os.path.join(base_path, file_path))
        else:
            raise RuntimeError('Choose dataset to test MonoDEVSNet model')
        print('Total number of images in {} dataset: {}'.format(self.opt.dataset, len(self.im_path_list)))

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
        with torch.no_grad():
            for iter_l, im_path in tqdm(enumerate(self.im_path_list)):
                input_color, input_im_np = self.load_rgb_image(im_path)

                time_taken = time.time()
                features, _ = self.models["encoder"](input_color)
                output = self.models["depth_decoder"](features)
                time_taken -= time.time()
                time_for_each_frame.append(np.abs(time_taken))

                # Convert disparity into depth maps
                pred_disp = self.invdepth_to_depth(output[("disp", 0)])
                pred_disp = pred_disp[0, 0].cpu().numpy()
                pred_depth_raw = 3. / pred_disp.copy()

                if self.opt.dataset == 'kitti':
                    gt_depth = self.gt_depths[iter_l]
                    gt_height, gt_width = gt_depth.shape
                    pred_disp = cv2.resize(pred_disp, (gt_width, gt_height), cv2.INTER_NEAREST)
                    pred_depth = 3. / pred_disp.copy()

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

            if self.opt.dataset == 'kitti':
                errors_absolute = np.array(errors_absolute).mean(0)

                print('\n  \n  for 80 meters - MonoDEVSNet Absolute depth estimation results '
                      'fps: {}'.format(1 / np.mean(time_for_each_frame)))
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

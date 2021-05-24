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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
example command line arguments:
 python3 monodevnset_trainer.py --cuda_idx 0 --num_workers 0 --batch_size 10 --height 192 --width 640 --max_depth 80 \
 --use_dc --use_le --use_ms --version markX --num_epochs 200 \
 --real_dataset kitti --syn_dataset vk_2.0 --real_data_path /mnt/largedisk/Datasets/KITTI \
 --syn_data_path /mnt/largedisk/Datasets
"""

import json
import os
import shutil
import sys
import time
from copy import deepcopy
from os.path import expanduser

import cv2
import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

import networks
import monodepth2
from datasets_EXT import VK1Dataset, VK2Dataset
from monodepth2 import KITTIRAWDataset, KITTIDepthDataset, KITTIOdomDataset
from monodepth2.evaluate_depth import compute_errors
from monodepth2.layers import disp_to_depth, compute_depth_errors
from monodepth2.trainer import Trainer
from monodepth2.utils import normalize_image, readlines
from utils import get_n_params
from utils.monodevsnet_options import MonoDEVSOptions


class MonoDEVSNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MonoDEVSNetTrainer, self).__init__(*args, **kwargs)

        # Load experiments options/parameters
        self.opt.trainer_name = 'MonoDEVSNetTrainer'
        self.opt.use_pose_net = self.use_pose_net

        # Set cuda index
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.opt.cuda_idx)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Remove unnecessary variables from self object
        for attr in ('models', 'model_optimizer', 'parameters_to_train', 'model_lr_scheduler',
                     'dataset', 'train_loader', 'val_loader', 'val_iter'):
            self.__dict__.pop(attr, None)
        self.models, self.model_optimizer, self.parameters_to_train = {}, {}, []

        # Load network architecture
        ''' Setting Models, initialization and optimization '''
        self.torch_zero = torch.tensor(0.).float().to(self.device)
        self.True_ = torch.tensor(np.ones(self.opt.batch_size)).float().to(self.device)
        self.False_ = torch.tensor(np.zeros(self.opt.batch_size)).float().to(self.device)
        with open(self.opt.config, 'r') as cfg:
            self.config = yaml.safe_load(cfg)

        # Encoder for (depth, segmentation, pose)
        self.models["encoder"] = self.network_selection('encoder')
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # Depth decoder
        self.models["depth_decoder"] = self.network_selection('depth_decoder')
        self.parameters_to_train += list(self.models["depth_decoder"].parameters())

        # Pose Encoder and Decoder
        if self.opt.use_pose_net:
            self.models["pose_encoder"] = self.network_selection('pose_encoder')
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            self.models["pose"] = self.network_selection('pose')
            self.parameters_to_train += list(self.models["pose"].parameters())

        # Domain classifier
        if self.opt.use_dc:
            self.models["domain_classifier"] = self.network_selection('domain_classifier')
            self.parameters_to_train += list(self.models["domain_classifier"].parameters())

        # Set optimization parameters
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Train model on device : {} \n  ", self.device)
        print("number of training parameters for each model")
        for model_name, model in self.models.items():
            print('{:^15}: {:^15}: {:>5.2f} M'.format(model_name, self.opt.models_fcn_name[model_name],
                                                      get_n_params(model) / 1000000))

        # Load dataloader
        # Setting Datasets
        img_ext = '.png' if self.opt.png else '.jpg'

        self.syn_or_real = ''
        datasets_dict = {"kitti": KITTIRAWDataset,
                         "kitti_odom": KITTIOdomDataset,
                         "kitti_depth": KITTIDepthDataset,
                         "vk_1.0": VK1Dataset,
                         "vk_2.0": VK2Dataset}
        self.real_dataset = datasets_dict[self.opt.real_dataset]
        self.syn_dataset = datasets_dict[self.opt.syn_dataset]

        syn_f_path = os.path.join(os.path.dirname(__file__), "splits", self.opt.syn_dataset, "train_files.txt")
        syn_train_dataset = self.syn_dataset(self.opt, csv_file_path=syn_f_path, frame_ids=[0],
                                             num_scales=4, is_train=True)

        real_f_path = os.path.join(os.path.dirname(__file__), "monodepth2/splits", self.opt.split, "{}_files.txt")
        real_filenames = readlines(real_f_path.format("train"))
        real_train_dataset = self.real_dataset(data_path=self.opt.real_data_path, filenames=real_filenames,
                                               height=self.opt.height, width=self.opt.width,
                                               frame_idxs=self.opt.frame_ids, num_scales=4, is_train=True,
                                               img_ext=img_ext)

        # Validation data-set and data-loader
        syn_val_dataset = self.syn_dataset(self.opt, csv_file_path=syn_f_path, frame_ids=[0],
                                           num_scales=4, is_train=False)
        real_filenames = readlines(real_f_path.format("val"))
        real_val_dataset = self.real_dataset(data_path=self.opt.real_data_path, filenames=real_filenames,
                                             height=self.opt.height, width=self.opt.width,
                                             frame_idxs=self.opt.frame_ids, num_scales=4, is_train=False,
                                             img_ext=img_ext)

        real_f_path = os.path.join(os.path.dirname(__file__), "monodepth2/splits", "eigen", "{}_files.txt")
        real_eigen_filenames = readlines(real_f_path.format("test"))
        real_eigen_val_dataset = self.real_dataset(data_path=self.opt.real_data_path, filenames=real_eigen_filenames,
                                                   height=self.opt.height, width=self.opt.width,
                                                   frame_idxs=[0], num_scales=4, is_train=False, img_ext=img_ext)

        # Training data-loaders
        self.real_train_loader = DataLoader(
            real_train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.syn_train_loader = DataLoader(
            syn_train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        # Validation data-loaders
        self.real_val_loader = DataLoader(
            real_val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.syn_val_loader = DataLoader(
            syn_val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.real_eigen_val_loader = DataLoader(
            real_eigen_val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)

        # Training iteration approach
        self.real_train_iter, self.syn_train_iter = iter(self.real_train_loader), iter(self.syn_train_loader)
        # val iteration approach
        self.real_val_iter, self.syn_val_iter, self.real_eigen_val_iter = \
            iter(self.real_val_loader), iter(self.syn_val_loader), iter(self.real_eigen_val_loader)

        gt_path = os.path.join(os.path.dirname(__file__), 'splits', self.opt.eval_split, "gt_depths.npz")
        self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        self.num_total_steps = real_train_dataset.__len__() // self.opt.batch_size * self.opt.num_epochs
        self.num_total_batch = real_train_dataset.__len__() // self.opt.batch_size
        self.im_shape = real_train_dataset.full_res_shape

        self.opt.model_name = (self.opt.models_fcn_name['encoder'] + str(self.opt.num_layers) +
                               '_trainwith' + str(self.opt.train_with) +
                               '_sD' + str(self.opt.syn_dataset) +
                               '_rD' + str(self.opt.real_dataset) +
                               '_F' + str(self.opt.frame_ids) +
                               '_le' + str(self.opt.use_le)[0] +
                               '_dc' + str(self.opt.use_dc)[0] +
                               '_ms' + str(self.opt.use_ms)[0] +
                               '_' + self.opt.version).replace(' ', '')
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        print('log directory path: {}'.format(self.log_path))

        # Setting tensorboard
        self.writers = {}
        for mode in ["train", "val_real", "val_syn", "val_real_eigen", "val_real_kitti_2015"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # Additional loss functions 
        self.L1Loss = nn.L1Loss().to(self.device)
        self.L2Loss = nn.MSELoss().to(self.device)
        self.CrossEntropy = nn.CrossEntropyLoss().to(self.device)
        self.LossDomainClassifier = nn.NLLLoss().to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are real: {:d}, syn: {:d} training items and "
              "real: {:d}, real Eigen: {:d}, syn: {:d} validation items\n".
              format(len(real_train_dataset), len(syn_train_dataset),
                     len(real_val_dataset), len(real_eigen_val_dataset), len(syn_val_dataset)))

        # train def
        self.step, self.epoch, self.previous_sp_loss = 0, 0, 0
        self.early_phase, self.mid_phase, self.late_phase = False, False, False
        self.start_time = time.time()

        self.save_opts()

    def train(self):
        # Function to save the best model
        def save_best_model(self_l):
            save_path = None
            save_folder = os.path.join(self_l.log_path, "models", "best_epoch_rsf_{}".format(best_epoch_rsf))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for model_name_l, model_dict in best_model_weights_rsf.items():
                save_path = os.path.join(save_folder, "{}.pth".format(model_name_l))
                if model_name_l == 'encoder':
                    # save the sizes - these are needed at prediction time
                    model_dict['height'] = self_l.opt.height
                    model_dict['width'] = self_l.opt.width
                    model_dict['use_stereo'] = self_l.opt.use_stereo
                    model_dict['best_model_mean_errors'] = torch.tensor(best_model_mean_errors_rsf)
                    model_dict['best_epoch'] = torch.tensor(best_epoch_rsf)
                    model_dict['best_abs_rel'] = best_abs_rel_rsf
                torch.save(model_dict, save_path)

            save_folder = os.path.join(self_l.log_path, "models", "best_epoch_asf_{}".format(best_epoch_asf))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for model_name_l, model_dict in best_model_weights_asf.items():
                save_path = os.path.join(save_folder, "{}.pth".format(model_name_l))
                if model_name_l == 'encoder':
                    # save the sizes - these are needed at prediction time
                    model_dict['height'] = self_l.opt.height
                    model_dict['width'] = self_l.opt.width
                    model_dict['use_stereo'] = self_l.opt.use_stereo
                    model_dict['best_model_mean_errors'] = torch.tensor(best_model_mean_errors_asf)
                    model_dict['best_epoch'] = torch.tensor(best_epoch_asf)
                    model_dict['best_abs_rel'] = best_abs_rel_asf
                torch.save(model_dict, save_path)
            if save_path is None:
                print('Could n\'t save the models. It has to run at least for one epoch')
            else:
                print('saving the models @ {}'.format(save_path))
                print('Best model weights are saved!')

        # initialize everything with worst possible results
        best_abs_rel_rsf, best_epoch_rsf, best_model_weights_rsf, best_model_mean_errors_rsf = 100.0, 0, {}, []
        best_abs_rel_asf, best_epoch_asf, best_model_weights_asf, best_model_mean_errors_asf = 100.0, 0, {}, []

        # Run entire pipeline
        try:
            for self.epoch in range(self.opt.num_epochs):
                # Zeros optimization on all model_optimizer
                self.zero_grad()
                self.run_epoch()

                # Run and extract results on validation KITTI Eigen split
                tt_val = time.time()
                if True:
                    # Evaluation on KITTI Eigen Validation set dataset
                    mean_errors_rsf, mean_errors_asf = self.val_real_eigen_dataset()

                    # Save best model based on best relative depth
                    best_model_mean_errors_rsf.append(mean_errors_rsf)
                    if mean_errors_rsf[0] < best_abs_rel_rsf:
                        best_epoch_rsf = self.epoch
                        for model_name, model in self.models.items():
                            model_state_dict = deepcopy(model.state_dict())
                            best_model_weights_rsf[model_name] = model_state_dict
                        best_abs_rel_rsf = mean_errors_rsf[0]

                    # Save best model based on best absolute depth
                    best_model_mean_errors_asf.append(mean_errors_asf)
                    if mean_errors_asf[0] < best_abs_rel_asf:
                        best_epoch_asf = self.epoch
                        for model_name, model in self.models.items():
                            model_state_dict = deepcopy(model.state_dict())
                            best_model_weights_asf[model_name] = model_state_dict
                        best_abs_rel_asf = mean_errors_asf[0]

                    # Evaluation on KITTI 2015 Training set dataset
                    # mean_errors_depth_kitti_2015, IoU_kitti_2015 = self.val_real_kitti_2015_dataset()

                if self.epoch % self.opt.save_frequency == 0:
                    self.save_model()
                tt_val = time.time() - tt_val
                print("\nTime taken to compute depth results on validation set: {} mins\n ".format(tt_val / 60))
            self.save_model()
            save_best_model(self)

        except (KeyboardInterrupt, SystemExit) as e:
            save_best_model(self)
            print(e)
            sys.exit(0)

    def zero_grad(self):
        self.model_optimizer.zero_grad()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()

        for batch_idx in range(0, self.num_total_batch):

            before_op_time = time.time()
            # Choosing the dataloader for training model
            if self.choosing_dataset_to_train_with(batch_idx):
                # Synthetic dataset
                self.syn_or_real = 'syn'
                try:
                    inputs = self.syn_train_iter.__next__()
                except StopIteration:
                    print('Stopped as the iteration has reached to the END, and reloading the synthetic dataloader')
                    self.syn_train_iter = iter(self.syn_train_loader)
                    inputs = self.syn_train_iter.__next__()
            else:
                # Real dataset
                self.syn_or_real = 'real'
                try:
                    inputs = self.real_train_iter.__next__()
                except StopIteration:
                    print('Stopped as the iteration has reached to the END, and reloading the real dataloader')
                    self.real_train_iter = iter(self.real_train_loader)
                    inputs = self.real_train_iter.__next__()

            # Move all available tensors to GPU memory
            for key, ipt in inputs.items():
                if type(key) == tuple or key == "depth_gt":
                    inputs[key] = ipt.to(self.device)

            # log less frequently after the first 2000 steps to save time & disk space
            self.step += 1
            self.early_phase = batch_idx % self.opt.log_frequency == 0
            self.mid_phase = False and self.step % self.opt.save_frequency == 0
            self.late_phase = self.num_total_batch - 1 == batch_idx

            outputs, losses = {}, {}
            # Depth estimation
            outputs_d, losses_d = self.process_batch(inputs)
            outputs.update(outputs_d)
            losses.update(losses_d)

            # No more if else conditions, just combine all losses based on availability of gradients
            final_loss = torch.tensor(0.).to(self.device)
            for k, v in losses.items():
                if ('d_' not in k) and v.requires_grad and ('/' not in k):
                    final_loss += v
            final_loss.backward()

            if (batch_idx + 1) % 2 == 0:
                self.model_optimizer.step()
                self.model_optimizer.zero_grad()
                self.zero_grad()

            duration = time.time() - before_op_time
            self.log_time(batch_idx, duration, losses["loss"].cpu().data)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            if self.early_phase or self.mid_phase or self.late_phase:
                self.log("train", inputs, outputs, losses)
                self.val("real")
                self.val("syn")

            if (batch_idx + 1) % 2 == 0:
                current_lr = self.update_learning_rate(self.model_optimizer, self.opt.learning_rate)

    # Depth Maps, Semantic Segmentation and Relative Pose Estimation
    def process_batch(self, inputs):
        """Pass a mini-batch through the network and generate images and losses
        """
        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        features, raw_hrnet_features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth_decoder"](features)

        if self.opt.use_dc:
            lambda_ = 1.0
            outputs['domain_classifier'] = self.models['domain_classifier'](raw_hrnet_features, lambda_)

        if self.opt.use_pose_net and "real" in self.syn_or_real:
            outputs.update(self.predict_poses(inputs, features))

        # convert estimated disparity from neural network to depth
        self.generate_images_pred_local(inputs, outputs)

        # loss functions
        losses = self.compute_losses_local(inputs, outputs)

        return outputs, losses

    def val(self, subset):
        """Validate the model on a single mini-batch
        """
        self.set_eval()
        if "syn" in subset:
            try:
                inputs = self.syn_val_iter.__next__()
            except StopIteration:
                print('Stopped as the iteration has reached to the END, and reloading the dataloader')
                self.syn_val_iter = iter(self.syn_val_loader)
                inputs = self.syn_val_iter.__next__()
            self.syn_or_real = 'syn'
        elif "real" in subset:
            try:
                inputs = self.real_val_iter.__next__()
            except StopIteration:
                print('Stopped as the iteration has reached to the END, and reloading the dataloader')
                self.real_val_iter = iter(self.real_val_loader)
                inputs = self.real_val_iter.__next__()
            self.syn_or_real = 'real'
        else:
            raise RuntimeError("Need proper validation loader choose syn or real or real_eigen")

        # Move all available tensors to GPU memory
        for key, ipt in inputs.items():
            if type(key) == tuple or key == "depth_gt":
                inputs[key] = ipt.to(self.device)

        outputs, losses = {}, {}
        with torch.no_grad():
            # Estimated depth and segmentation
            outputs_d, losses_d = self.process_batch(inputs)
            outputs.update(outputs_d)
            losses.update(losses_d)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val_" + self.syn_or_real, inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred_local(self, inputs, outputs):
        if self.syn_or_real == "syn":
            for scale in self.opt.scales:
                disp = outputs[("disp", scale)]
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

                outputs[("depth", 0, scale)] = depth

        elif self.syn_or_real == "real":
            self.generate_images_pred(inputs, outputs)
        else:
            raise RuntimeError("choose synthetic or real data")

    def compute_losses_local(self, inputs, outputs):
        # Choose loss function based on input data
        self_loss, l1_loss = self.torch_zero.clone(), self.torch_zero.clone()
        losses = {"loss": self.torch_zero.clone()}

        # Domain classifier with Gradient Reversal Layer
        if self.opt.use_dc:
            if self.syn_or_real == 'syn':
                losses['domain_classifier'] = 10.0 * self.LossDomainClassifier(outputs['domain_classifier'],
                                                                               self.False_.long())
            else:
                losses['domain_classifier'] = 10.0 * self.LossDomainClassifier(outputs['domain_classifier'],
                                                                               self.True_.long())

            losses['loss/domain_classifier'] = losses['domain_classifier'].detach().cpu().data

        if 'self' in self.opt.real_loss_fcn and "real" in self.syn_or_real:
            losses.update(self.compute_losses(inputs, outputs))

            # Loss equalizer
            losses["loss/self"] = losses["loss"].cpu().data
            weight_self_loss = self.previous_sp_loss / losses["loss/self"] if self.opt.use_le else 1
            losses["loss"] *= weight_self_loss

        # Supervised loss function - Mostly for virtual kitti images as it has ground truth
        elif 'l1' in self.opt.syn_loss_fcn and "syn" in self.syn_or_real:
            l1_loss = self.supervised_loss(inputs, outputs)
            losses["loss"] += l1_loss
            self.previous_sp_loss = l1_loss.cpu().data
            losses["loss/l1_loss"] = l1_loss.cpu().data
        else:
            raise RuntimeError('choose a loss function for each syn and real dataset')

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > self.opt.min_depth) & (depth_gt < self.opt.max_depth)

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, depth_gt.shape[2:], mode="bilinear", align_corners=False), self.opt.min_depth,
            self.opt.max_depth)
        depth_pred = depth_pred.detach()

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        median_gt_pred = torch.median(depth_gt) / torch.median(depth_pred)
        depth_pred *= median_gt_pred

        depth_pred = torch.clamp(depth_pred, min=self.opt.min_depth, max=self.opt.max_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            if 'median' in metric:
                losses[metric] = median_gt_pred
            else:
                losses[metric] = depth_errors[i]

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_so_far = time.time() - self.start_time
        print_string = "exp_name {}  \n| dataset: {:>5} | epoch {:>3} | batch {:>6}/{:>6} | " \
                       "examples/s: {:5.1f} | loss: {:.5f}"
        print(print_string.format(self.log_path.split('/')[-1], self.syn_or_real, self.epoch, batch_idx,
                                  self.num_total_batch, samples_per_sec, loss))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            if '/' in l:
                writer.add_scalar("{}".format(l), v, self.step)

        for j in [0]:  # range(min(4, self.opt.batch_size)):  # write a maximum of four images
            writer.add_image(
                "depth_gt_{}/{}".format(0, j),
                normalize_image(inputs["depth_gt"][j]), self.step)

            writer.add_image(
                "depth_pred_{}/{}".format(0, j),
                normalize_image(outputs[("depth", 0, 0)][j]), self.step)

            diff = torch.abs(F.interpolate(inputs["depth_gt"] / self.opt.syn_scaling_factor,
                                           [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)[j]
                             - outputs[("depth", 0, 0)][j])
            mask = F.interpolate((inputs["depth_gt"][j] > 0).float().unsqueeze(0), diff.shape[1:]).squeeze()
            diff = diff * mask.float()
            writer.add_image(
                "abs_depth_diff_{}/{}".format(0, j),
                normalize_image(diff), self.step)

            for s in [0]:  # self.opt.scales:
                frame_ids = [0]  # For time being

                for frame_id in frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                if 'self' in self.opt.loss_fcn:
                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", s)][j]), self.step)

                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        # save code as another folder in log_path
        dst_path = os.path.join(self.log_path, 'code', 'v0')
        iter_yes_or_no = 0
        while os.path.exists(dst_path):
            dst_path = os.path.join(self.log_path, 'code', 'v' + str(iter_yes_or_no))
            iter_yes_or_no = iter_yes_or_no + 1
        user_name = expanduser("~")
        try:
            shutil.copytree(os.getcwd(), dst_path, ignore=shutil.ignore_patterns('*.pyc', 'tmp*'))
        except Exception as e_copytree:
            print(e_copytree)

        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w', encoding='utf-8') as f:
            json.dump(to_save, f, indent=2)

    def update_learning_rate(self, optimizer, base_lr_rate, power=0.9):
        step = max(1, self.step)
        step = int(step / 2)
        new_lr_rate = base_lr_rate * (1 - (float(step / self.num_total_steps) ** power))
        optimizer.param_groups[0]['lr'] = new_lr_rate

        return new_lr_rate

    # Train with - syn - 0, real - 1, both - 2
    def choosing_dataset_to_train_with(self, index):
        if self.opt.train_with == "syn":
            return True
        elif self.opt.train_with == "real":
            return False
        elif self.opt.train_with == "both":
            if index % 2 == 0:
                return True
            else:
                return False
        else:
            print("choose at-least one of the dataset for training")
            sys.exit(0)

    def supervised_loss(self, inputs, outputs):
        l1_loss = self.torch_zero.clone()
        for i in self.opt.scales:
            pred = outputs[("depth", 0, i)]
            gt = F.interpolate(inputs["depth_gt"], size=pred.shape[2:]) / self.opt.syn_scaling_factor

            mask = torch.zeros(gt.shape, dtype=torch.float).to(self.device)
            mask[gt < (self.opt.max_depth - 2) / self.opt.syn_scaling_factor] = 1
            mask[gt == 0] = 0  # while we train with kitti LiDAR GT

            if self.opt.use_ms:
                mask_segm = torch.zeros(gt.shape, dtype=torch.float).to(self.device)
                mask_segm[(inputs[('segm_gt', 0, 0)] == 8)] = 1.0
                mask_segm[(inputs[('segm_gt', 0, 0)] == 9)] = 1.0
                mask_segm[(inputs[('segm_gt', 0, 0)] == 10)] = 1.0
                mask_segm[(inputs[('segm_gt', 0, 0)] == 11)] = 1.0
                mask_segm[(mask_segm == 0)] = 0.5
                mask = mask * mask_segm

            l1_loss += torch.mean(torch.abs(pred * mask - gt * mask))

        return l1_loss

    def val_real_eigen_dataset(self):
        """Validate the model on a single mini-batch
        """
        self.set_eval()

        pred_disps = []

        for batch_idx, inputs in enumerate(self.real_eigen_val_loader, 0):

            # Move all available tensors to GPU memory
            for key, ipt in inputs.items():
                if type(key) == tuple or key == "depth_gt":
                    inputs[key] = ipt.to(self.device)

            outputs, losses = {}, {}
            with torch.no_grad():
                features, raw_hrnet_features = self.models["encoder"](inputs["color_aug", 0, 0])
                outputs = self.models["depth_decoder"](features)
                pred_disp, _ = disp_to_depth(outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

        errors_rsf, errors_asf = [], []
        ratios = []
        pred_depths_copy, gt_depths_copy = [], []
        for i in range(pred_disps.shape[0]):

            gt_depth = self.gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            if self.opt.eval_split == "eigen":
                mask = np.logical_and(gt_depth > self.opt.min_depth, gt_depth < self.opt.max_depth)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
            else:
                mask = gt_depth > 0

            pred_depth_asf = pred_depth.copy() * self.opt.syn_scaling_factor
            ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
            ratios.append(ratio)
            pred_depth_rsf = pred_depth.copy() * ratio

            # Create a copy
            pred_depths_copy.append(np.expand_dims(pred_depth_asf.copy(), axis=0))
            gt_depths_copy.append(np.expand_dims(gt_depth.copy(), axis=0))

            # Choose only valid pixels - using mask
            pred_depth_rsf = pred_depth_rsf[mask]
            pred_depth_asf = pred_depth_asf[mask]
            gt_depth = gt_depth[mask]

            # Clamping the min and max depth
            pred_depth_rsf[pred_depth_rsf < self.opt.min_depth] = self.opt.min_depth
            pred_depth_rsf[pred_depth_rsf > self.opt.max_depth] = self.opt.max_depth
            pred_depth_asf[pred_depth_asf < self.opt.min_depth] = self.opt.min_depth
            pred_depth_asf[pred_depth_asf > self.opt.max_depth] = self.opt.max_depth

            # Compute depth metric error values for individual and constant scaling factor
            errors_rsf.append(compute_errors(gt_depth, pred_depth_rsf))
            errors_asf.append(compute_errors(gt_depth, pred_depth_asf))

        print("\n \n KITTI Eigen Validation Split")
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
        med_std = [med, np.std(ratios / med)]

        mean_errors_rsf = np.array(errors_rsf).mean(0)
        mean_errors_asf = np.array(errors_asf).mean(0)
        print("\n rsf \n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors_rsf.tolist()) + "\\\\")
        print("\n asf \n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors_asf.tolist()) + "\\\\")

        self.log_evaluation('val_real_eigen', mean_errors_rsf, med_std, mean_errors_asf)
        self.set_train()

        return mean_errors_rsf, mean_errors_asf

    def log_evaluation(self, mode, errors_rsf, med_std, errors_asf):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]

        writer.add_scalar("median/mean_median", med_std[0], self.epoch)
        writer.add_scalar("median/median_std", med_std[1], self.epoch)

        writer.add_scalar("depth_rsf/abs_rel", errors_rsf[0], self.epoch)
        writer.add_scalar("depth_rsf/sq_rel", errors_rsf[1], self.epoch)
        writer.add_scalar("depth_rsf/rmse", errors_rsf[2], self.epoch)
        writer.add_scalar("depth_rsf/rmse_log", errors_rsf[3], self.epoch)
        writer.add_scalar("depth_rsf/a1", errors_rsf[4], self.epoch)
        writer.add_scalar("depth_rsf/a2", errors_rsf[5], self.epoch)
        writer.add_scalar("depth_rsf/a3", errors_rsf[6], self.epoch)
        writer.add_scalar("depth_asf/abs_rel", errors_asf[0], self.epoch)
        writer.add_scalar("depth_asf/sq_rel", errors_asf[1], self.epoch)
        writer.add_scalar("depth_asf/rmse", errors_asf[2], self.epoch)
        writer.add_scalar("depth_asf/rmse_log", errors_asf[3], self.epoch)
        writer.add_scalar("depth_asf/a1", errors_asf[4], self.epoch)
        writer.add_scalar("depth_asf/a2", errors_asf[5], self.epoch)
        writer.add_scalar("depth_asf/a3", errors_asf[6], self.epoch)

    # need to keep on updating based on requirement
    # Case sensitive
    def network_selection(self, model_key):
        if model_key == 'encoder':
            if 'HRNet' == self.opt.models_fcn_name[model_key]:
                return networks.HRNetPyramidEncoder(self.config).to(self.device)
            elif 'ResNet' == self.opt.models_fcn_name[model_key]:
                return networks.ResnetEncoder(self.opt.num_layers,
                                              self.opt.weights_init == "pretrained").to(self.device)
            else:
                raise RuntimeError('Choose a depth encoder within available scope')

        elif model_key == 'depth_decoder':
            if 'DepthDecoder' == self.opt.models_fcn_name[model_key]:
                return networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales).to(self.device)
            else:
                raise RuntimeError('Choose depth Decoder within available scope')

        # Add multiple domain classifiers
        elif model_key == 'domain_classifier':
            return networks.DomainClassifier(in_channel=720, width=self.opt.width,
                                             height=self.opt.height).to(self.device)

        elif model_key == 'pose_encoder':
            return networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=2).to(self.device)

        # Add other models
        elif model_key == 'pose':
            if self.opt.pose_model_type == "separate_resnet":
                return networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2).to(self.device)

            elif self.opt.pose_model_type == "shared":
                return networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames).to(self.device)

            elif self.opt.pose_model_type == "posecnn":
                return networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2).to(self.device)
            else:
                raise RuntimeError('Choose a depth encoder within available scope')

        else:
            raise RuntimeError('Don\'t forget to mention what you want!')


if __name__ == "__main__":
    # Load options
    opts = MonoDEVSOptions().parse()

    # Load MonoDEVSNet trainer scripts and start training
    monodevs = MonoDEVSNetTrainer(options=opts)
    monodevs.train()

    TheEnd = 1

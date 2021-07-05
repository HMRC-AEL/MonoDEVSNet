# Author: Akhil Gurram
# Build on top of the monodepth2
# (Automatically pulled from git repo, monodepth2 source code is not included in this repository)
# This is the training script of the MonoDEVSNet framework.
# MonoDEVSNet: Monocular Depth Estimation through Virtual-world Supervision and Real-world SfM Self-Supervision
# https://arxiv.org/abs/2103.12209


from __future__ import absolute_import, division, print_function

import os
import random
from abc import ABC

import kornia
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from monodepth2.datasets.mono_dataset import pil_loader


class DepthDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        opt
        csv_file_path
        num_scales
        is_train
    """

    def __init__(self,
                 opt,
                 csv_file_path,
                 frame_ids,
                 num_scales,
                 subset='a0',
                 is_train=False):
        super(DepthDataset, self).__init__()

        def shortlist_filenames():

            angles = []
            if self.ax == 'aall':
                angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
            elif self.ax == 'a0':
                angles = [0, 180]
            elif self.ax == 'a1':
                angles = [45, 135]
            elif self.ax == 'a2':
                angles = [90, 270]
            elif self.ax == 'a3':
                angles = [225, 315]
            else:
                print('choose subset among: aall, a0, a1, a2, a3')

            self.mono_or_stereo = 'mono'  # opt.training_cam_type
            try:
                training_cam_type = 'mono'  # opt.training_cam_type
                testing_cam_type = 'mono'  # opt.testing_cam_type

                list_of_angles_indexes = self.filenames['angle'].to_numpy()
                list_of_cam_type_indexes = self.filenames['mono_or_stereo'].to_numpy()
                list_of_train_or_val_indexes = self.filenames['train_or_val'].to_numpy()
                list_of_valid_or_not_indexes = self.filenames['valid_or_not'].to_numpy()
                list_of_depth_init_indexes = self.filenames['depth_starting'].to_numpy()
                list_subset_ax = self.filenames['ax'].to_numpy()

                if is_train:
                    train_or_val = 'train'
                    cam_type = training_cam_type
                    self.mono_or_stereo = training_cam_type
                else:
                    train_or_val = 'val'
                    cam_type = testing_cam_type
                    self.mono_or_stereo = testing_cam_type

                # based on angles
                set_indexes_angle = []
                for local_angle in angles:
                    set_indexes_angle = set_indexes_angle + np.where(list_of_angles_indexes == local_angle)[0].tolist()

                # based on cam type
                set_indexes_mono_or_stereo = np.where(list_of_cam_type_indexes == cam_type)[0].tolist()
                if 'mono' in training_cam_type:  # if its mono use stereo images also
                    set_indexes_mono_or_stereo = set_indexes_mono_or_stereo + \
                                                 np.where(list_of_cam_type_indexes == 'stereo')[0].tolist()

                # based on training or val set
                set_indexes_train_or_val = np.where(list_of_train_or_val_indexes == train_or_val)[0].tolist()

                # based on valid or not columns info :: valid = 1 else 0
                set_indexes_valid_or_not = np.where(list_of_valid_or_not_indexes == 1)[0].tolist()

                # based on depth values should start from 3 meters as minimum
                set_indexes_depth_init = np.where(list_of_depth_init_indexes > self.depth_init)[0].tolist()

                # based on depth values should start from 3 meters as minimum
                set_indexes_ax = []
                if self.ax == 'aall':
                    for ax_ in ['a0', 'a1', 'a2', 'a3', 'aall']:
                        set_indexes_ax += np.where(list_subset_ax == ax_)[0].tolist()
                else:
                    set_indexes_ax = np.where(list_subset_ax == self.ax)[0].tolist()

                # find common intersection of conditions nodes
                set_indexes_shortlisted = list(set(set_indexes_angle) &
                                               set(set_indexes_mono_or_stereo) &
                                               set(set_indexes_train_or_val) &
                                               set(set_indexes_valid_or_not) &
                                               set(set_indexes_depth_init) &
                                               set(set_indexes_ax))

                self.filenames = self.filenames.copy().reindex(set_indexes_shortlisted).reset_index()

            except Exception as e_not_:
                print(e_not_)
                self.filenames = self.filenames.copy()

            if opt.total_number_of_images_for_training == -1:
                self.filenames = self.filenames.copy()
            else:
                self.filenames = self.filenames[:opt.total_number_of_images_for_training]

            return self.filenames

        self.num_scales = num_scales
        self.is_train = is_train
        self.frame_ids = frame_ids

        self.opt = opt
        self.dataset_name = self.opt.syn_dataset
        if self.opt.syn_data_path is not None:
            self.data_path = self.opt.syn_data_path
        else:
            self.data_path = '/mnt/ssd1/Datasets'
        self.n_class = self.opt.n_class
        self.height = self.opt.height
        self.width = self.opt.width
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.do_flip = self.opt.do_flip
        self.do_color_aug = self.opt.do_color_aug

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=Image.ANTIALIAS)

        # from virtual KITTI dataset
        self.debug = False
        self.filenames = pd.read_csv(csv_file_path, delimiter=',')
        self.depth_init = self.opt.depth_init
        self.ax = subset

        self.filenames = shortlist_filenames()
        self.total_num_ims = self.filenames.shape[0]

        # iter, rgb, depth, Segmentation, street_address, weather, view, angle, dataset
        self.iter = 0
        self.street_address = ''
        self.weather = ''
        self.view = ''
        self.angle = ''
        self.shape = np.array(Image.open(os.path.join(self.data_path, self.filenames['rgb_l'][0]))).shape[:2]
        self.load_depth = True
        self.load_segm = True

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {"syn_or_real": "syn"}

        do_color_aug = self.is_train and self.do_color_aug and random.random() > 0.5
        do_flip = self.is_train and self.do_flip and random.random() > 0.5

        file_path = self.filenames['rgb_l'][index]
        side = self.filenames['l_or_r'][index]
        for i in self.frame_ids:
            if i == "s":
                other_side = {"r": ["/Right/", "/Left/"], "l": ["/Left/", "/Right/"]}[side]
                file_path = file_path.replace(other_side[0], other_side[1])
            inputs[("color", i, -1)] = self.get_color(file_path, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_ids:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            file_path = self.filenames['depth_l'][index]
            depth_gt = self.get_depth(file_path, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = F.interpolate(
                torch.from_numpy(inputs["depth_gt"].astype(np.float32)).unsqueeze(0),
                (self.height, self.width)).squeeze(0)

        if self.load_segm:
            file_path = self.filenames['segm_l'][index]
            segm_gt = self.get_segm(file_path, do_flip)
            if self.n_class == 2:
                mask_segm = np.ones(segm_gt.shape, dtype=np.int32)
                mask_segm[segm_gt == 8] = 0
                mask_segm[segm_gt == 9] = 0
                mask_segm[segm_gt == 10] = 0
                mask_segm[segm_gt == 11] = 0
                inputs["segm_gt", 0, 0] = np.expand_dims(mask_segm, 0)
            else:
                inputs["segm_gt", 0, 0] = np.expand_dims(segm_gt, 0)

            inputs["segm_gt", 0, 0] = F.interpolate(
                torch.from_numpy(inputs["segm_gt", 0, 0].astype(np.float32)).unsqueeze(0),
                (self.height, self.width)).long().squeeze(0)

            # semantic edges
            edges = kornia.laplacian(inputs["segm_gt", 0, 0].unsqueeze(0).float(), kernel_size=5).squeeze(0)
            inputs[("segm_edges", 0, 0)] = (edges[0] > 0.1).long()

        if "s" in self.frame_ids:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        edges = kornia.laplacian(inputs["color_aug", 0, 0].unsqueeze(0).float(), kernel_size=5)
        edges = edges / edges.max()
        inputs[("edges", 0, 0)] = (edges[0, 0] > 0.1).long()

        return inputs

    def get_color(self, file_path, do_flip):
        raise NotImplementedError

    def get_depth(self, file_path, do_flip):
        raise NotImplementedError

    def get_segm(self, file_path, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError


class VK1Dataset(DepthDataset, ABC):
    """Superclass for different types of Virtual KITTI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(VK1Dataset, self).__init__(*args, **kwargs)

        self.full_res_shape = (1242, 375)
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.K_orig = self.K.copy()

        self.side_map = {"l": '/Camera_0/', "r": '/Camera_1/'}
        self.VK_id_to_trainid = {-1: [0, 0, 0], 0: [90, 200, 255], 1: [140, 140, 140], 2: [255, 255, 0],
                                 3: [200, 200, 0], 4: [100, 60, 100], 6: [210, 0, 200], 7: [255, 127, 80],
                                 8: [160, 60, 60], 9: [0, 139, 139], 11: [0, 199, 0], 11.1: [90, 240, 0],
                                 12: [80, 80, 80], 13: [255, 130, 0], 15: [250, 100, 255]}

    def get_color(self, file_path, do_flip):
        color = Image.open(os.path.join(self.data_path, file_path)).convert('RGB')
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        color = np.array(color, dtype=np.float32)

        return Image.fromarray(color.astype(np.uint8))

    def get_depth(self, file_path, do_flip, h_matrix=None, wanted_crop=None):
        depth = Image.open(os.path.join(self.data_path, file_path)).convert('I')
        if do_flip:
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        # normalize depth values
        depth_np = np.array(depth, dtype=np.float32)
        depth_np = depth_np / 100
        depth_np[depth_np > self.opt.max_depth] = self.opt.max_depth
        depth_np_orig = depth_np.copy()

        return depth_np

    def get_segm(self, file_path, do_flip, h_matrix=None, wanted_crop=None):
        # Virtual kitti 1.3 dataset
        segm = Image.open(os.path.join(self.data_path, file_path)).convert('RGB')
        if do_flip:
            segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

        segm = np.array(segm, dtype=np.int32)
        segm_copy = np.zeros(segm.shape[0:2]) - 1
        for v, [k0, k1, k2] in self.VK_id_to_trainid.items():
            valid_mask = np.logical_and(np.logical_and(segm[:, :, 0] == k0, segm[:, :, 1] == k1),
                                        segm[:, :, 2] == k2)
            segm_copy[valid_mask] = int(v)
        segm = segm_copy.astype(np.int32) + 1
        segm[segm > 16] = 0

        return segm.astype(np.long)


class VK2Dataset(DepthDataset, ABC):
    """Superclass for different types of Virtual KITTI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(VK2Dataset, self).__init__(*args, **kwargs)

        self.full_res_shape = (1242, 375)
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.K_orig = self.K.copy()

        self.side_map = {"l": '/Camera_0/', "r": '/Camera_1/'}
        self.VK_id_to_trainid = {-1: [0, 0, 0], 0: [90, 200, 255], 1: [140, 140, 140], 2: [255, 255, 0],
                                 3: [200, 200, 0], 4: [100, 60, 100], 6: [210, 0, 200], 7: [255, 127, 80],
                                 8: [160, 60, 60], 9: [0, 139, 139], 11: [0, 199, 0], 11.1: [90, 240, 0],
                                 12: [80, 80, 80], 13: [255, 130, 0], 15: [250, 100, 255]}

    def get_color(self, file_path, do_flip):
        color = Image.open(os.path.join(self.data_path, file_path)).convert('RGB')
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        color = np.array(color, dtype=np.float32)

        return Image.fromarray(color.astype(np.uint8))

    def get_depth(self, file_path, do_flip, h_matrix=None, wanted_crop=None):
        depth = Image.open(os.path.join(self.data_path, file_path)).convert('I')
        if do_flip:
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        # normalize depth values
        depth_np = np.array(depth, dtype=np.float32)
        depth_np = depth_np / 100
        depth_np[depth_np > self.opt.max_depth] = self.opt.max_depth
        depth_np_orig = depth_np.copy()

        return depth_np

    def get_segm(self, file_path, do_flip, h_matrix=None, wanted_crop=None):
        # Virtual kitti 1.3/2.0 dataset
        segm = Image.open(os.path.join(self.data_path, file_path)).convert('RGB')
        if do_flip:
            segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

        segm = np.array(segm, dtype=np.int32)
        segm_copy = np.zeros(segm.shape[0:2]) - 1
        for v, [k0, k1, k2] in self.VK_id_to_trainid.items():
            valid_mask = np.logical_and(np.logical_and(segm[:, :, 0] == k0, segm[:, :, 1] == k1),
                                        segm[:, :, 2] == k2)
            segm_copy[valid_mask] = int(v)
        segm = segm_copy.astype(np.int32) + 1
        segm[segm > 16] = 0

        return segm.astype(np.longlong)

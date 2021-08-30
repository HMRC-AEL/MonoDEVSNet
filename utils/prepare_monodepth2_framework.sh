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

# clone monodepth2 repository
cd ..
git clone https://github.com/nianticlabs/monodepth2

# Rearrange imports
# In kitti dataset folder
sed -i 's/from kitti_utils/from ..kitti_utils/g' monodepth2/datasets/kitti_dataset.py
sed -i 's/transforms.ColorJitter.get_params(/transforms.ColorJitter(/g' monodepth2/datasets/mono_dataset.py

# In evaluate depth file
sed -i 's/from layers/from monodepth2.layers/g' monodepth2/evaluate_depth.py
sed -i 's/from options/from monodepth2.options/g' monodepth2/evaluate_depth.py
sed -i 's/import datasets/from monodepth2 import datasets/g' monodepth2/evaluate_depth.py

# In monodepth2 trainer.py file
sed -i 's/from kitti_utils/from monodepth2.kitti_utils/g' monodepth2/trainer.py
sed -i 's/from layers/from monodepth2.layers/g' monodepth2/trainer.py
sed -i 's/import datasets/from monodepth2 import datasets/g' monodepth2/trainer.py
sed -i 's/"border", align_corners=False)/"border", align_corners=True)/g' monodepth2/trainer.py
sed -i 's/self.opt.num_layers/50/g' monodepth2/trainer.py

# In monodepth2 network folder
sed -i 's/from layers/from monodepth2.layers/g' monodepth2/networks/depth_decoder.py
sed -i 's/return self.features/return self.features, self.features\[-1\]/g'  monodepth2/networks/resnet_encoder.py
rm monodepth2/networks/__init__.py

# In monodepth2 kitti_utils.py file
sed -i 's/.astype(np.int)/.astype(np.int32)/g' monodepth2/kitti_utils.py

# In monodepth2 options.py file
sed -i 's/choices=\[18, 34, 50, 101, 152\]/choices=\[18, 32, 34, 48, 50, 101, 152, 121, 161, 169, 201\]/g' monodepth2/options.py
sed -i 's/choices=\["kitti",/choices=\["any", "kitti",/g' monodepth2/options.py
sed -i 's/default=100.0/default=80.0/g' monodepth2/options.py

# change __init__ file in monodepth2/network to exclude depth network
touch monodepth2/__init__.py
echo from .layers import SSIM, BackprojectDepth, Project3D >> monodepth2/__init__.py
echo from .options import MonodepthOptions >> monodepth2/__init__.py
echo from .datasets.mono_dataset import MonoDataset >> monodepth2/__init__.py
echo from .datasets.kitti_dataset import KITTIDataset, KITTIDepthDataset, KITTIOdomDataset, KITTIRAWDataset >> monodepth2/__init__.py

echo "# ready to go!"


import os
from distutils.dir_util import copy_tree

from tqdm import tqdm

# KITTI base path where to save the proj_depth files
save_base_path = '/mnt/largedisk/Datasets/KITTI'
# path of the extracted (KITTI) data_depth_annotated.zip files - train or val
load_base_path = '/mnt/largedisk/Datasets/KITTI/train'

for folder_name in tqdm(sorted(os.listdir(load_base_path))):
    base_folder_name = folder_name.split('_drive')[0]
    from_directory = os.path.join(load_base_path, folder_name, 'proj_depth')
    to_directory = os.path.join(save_base_path, base_folder_name, folder_name, 'proj_depth')
    copy_tree(from_directory, to_directory)

TheEnd = 1

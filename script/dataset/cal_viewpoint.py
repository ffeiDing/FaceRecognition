"""Refactor file directories, save/rename images and partition the 
train/val/test set, in order to support the unified dataset interface.
"""

from __future__ import print_function

import sys
sys.path.insert(0, '.')
import dlib
from zipfile import ZipFile
import os
import os.path as osp
import numpy as np

from tri_loss.utils.utils import may_make_dir
from tri_loss.utils.utils import save_pickle
from tri_loss.utils.utils import load_pickle

from tri_loss.utils.dataset_utils import get_im_names
from tri_loss.utils.dataset_utils import partition_train_val_set
from tri_loss.utils.dataset_utils import new_im_name_tmpl
from tri_loss.utils.dataset_utils import parse_im_name as parse_new_im_name
from tri_loss.utils.dataset_utils import move_ims
import pickle
import cv2
import random
ospj = osp.join
ospeu = osp.expanduser
import argparse


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/dff/f/shape_predictor_68_face_landmarks.dat")

def partitions(mask_paths_list, raw_path):
    mask = [0, 0, 0, 0, 0]
    ori = [0, 0, 0, 0, 0]
    #Test_set
    name_list = os.listdir(test_dataset_path)
    for id_name in name_list:
        id_name_path = os.path.join(test_dataset_path,id_name)
        if not os.path.isdir(id_name_path):
            continue
        id_name_list = os.listdir(id_name_path)
        is_mask = int(id_name.split('_')[-1])
        id = int(id_name.split('_')[-2])
        for im_name in id_name_list:
            cam = int(im_name.split('_')[-1].split('.')[0])
            if is_mask == 0:
                ori[cam] = ori[cam]+1
            else:
                mask[cam] = mask[cam]+1

    print(mask)
    print(ori)


      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    train_mask_dataset_paths = ['/ssd_datasets/dff/data/mask/new','/ssd_datasets/dff/data/mask/3m',
                      '/ssd_datasets/dff/data/mask/n95','/ssd_datasets/dff/data/mask/mask_1',
                      '/ssd_datasets/dff/data/mask/mask_2','/ssd_datasets/dff/data/mask/mask_3',
                      '/ssd_datasets/dff/data/mask/mask_4/','/ssd_datasets/dff/data/mask/mask_5']
    parser.add_argument('--train_ori_dataset_path', type=str, default="/home/dff/f/data/faces_webface_112x112_raw_image")
    parser.add_argument('--test_dataset_path', type=str, default="/ssd_datasets/dff/1018")
    args = parser.parse_args()
    test_dataset_path = args.test_dataset_path
    train_ori_dataset_path = args.train_ori_dataset_path
    partitions(train_mask_dataset_paths, train_ori_dataset_path)

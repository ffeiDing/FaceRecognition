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

def partitions(mask_paths_list, raw_path, save_path):
    new_trainval_im_names = []
    new_trainval_im_ids = []
    new_trainval_im_scales = []

    train_ids=[] #list(set(new_trainval_im_ids))

    train_ids2labels=[]#dict(zip(train_ids,range(len(train_ids))))
    
    #Test_set
    all_im_names = []
    all_im_ids = []
    all_im_masks = []
    all_im_cams = []
    name_list = os.listdir(test_dataset_path)
    for id_name in name_list:
        id_name_path = os.path.join(test_dataset_path,id_name)
        if not os.path.isdir(id_name_path):
            continue
        id_name_list = os.listdir(id_name_path)
        mask = int(id_name.split('_')[-1])
        id = int(id_name.split('_')[-2])
        for im_name in id_name_list:
            cam = int(im_name.split('_')[-1].split('.')[0])
            all_im_names.append(osp.join(id_name_path,im_name))
            all_im_ids.append(id)
            all_im_masks.append(mask)
            all_im_cams.append(cam)
            print(id, mask, cam)

    print('all_im_names',len(all_im_names))


    test_im_names=[]
    test_im_ids=[]
    test_marks=[]
    test_im_cams=[]    
    ##val is test
    q = 0
    g = 0
 
    for i in range(len(all_im_names)):
      # test_im_name = test_im_names[i]
        mask_id = all_im_masks[i] 
        im_id= all_im_ids[i]
        cam_id = all_im_cams[i]

        if mask_id==0:
            test_im_names.append(all_im_names[i])
            # print(test_im_names)
            test_im_ids.append(im_id)
            test_marks.append(0)
            test_im_cams.append(0)
            q = q+1
        
        if True:
            test_im_names.append(all_im_names[i])
            # print(test_im_names)
            test_im_ids.append(im_id)
            test_marks.append(1)
            test_im_cams.append(1)
            g = g+1
      

    new_partitions = {
        'trainval_im_names': new_trainval_im_names,
        'trainval_im_scales': new_trainval_im_scales,
        'trainval_im_ids': new_trainval_im_ids,
        'trainval_ids2labels': train_ids2labels,
        'test_marks': test_marks,
        'test_im_names':test_im_names,
        'test_ids': test_im_ids,
        'test_cams':test_im_cams
    }
    save_pickle(new_partitions, save_path)
    print('Partition file saved to {}'.format(save_path))
    print('trainval_im_names',len(new_trainval_im_ids))
    print('trainval_im_ids',len(np.unique(new_trainval_im_ids)))
    print('test_im_names',len(test_im_names))
    print('query',q)
    print('gallery',g)


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
    partition_path = osp.join("/home/dff/MFR/FaceRecognition/script/pkl", "test_normal_all.pkl")
    partitions(train_mask_dataset_paths, train_ori_dataset_path, partition_path)

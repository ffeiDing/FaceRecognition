
from __future__ import print_function

import sys
import warnings
import pickle
import cv2
from numpy.lib.function_base import extract
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic=True
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import json
from sklearn.cluster import KMeans, SpectralClustering

from tri_loss.dataset import create_dataset
from tri_loss.dataset.TestSet import TestSet
from tri_loss.model.model_stn_mask import Model
from tri_loss.model.loss import *
from tri_loss.utils.utils import *
from tri_loss.utils.utils import tight_float_str as tfs


def load_pkl(path):
    with open(path, 'rb') as inp:  # Overwrites any existing file.
        info_dict = pickle.load(inp)
    return info_dict


class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(3,))
        parser.add_argument('-r', '--run', type=int, default=1)
        parser.add_argument('--set_seed', type=str2bool, default=False)
        # parser.add_argument('--partition_path', type=str, default='/ssd_datasets/dff/1018/test_1018_partition.pkl')
        parser.add_argument('--partition_path', type=str, default='/home/dff/f/data/faces_webface_112x112_raw_image/mask_landmark.pkl')
        parser.add_argument('--trainset_part', type=str, default='trainval_landmark')
        parser.add_argument('--resize_h_w', type=eval, default=(128,128))#(128, 128))
        # These several only for training set
        parser.add_argument('--crop_prob', type=float, default=0.0)
        parser.add_argument('--crop_ratio', type=float, default=0.9)
        parser.add_argument('--rotate_prob', type=float, default=0.5)
        parser.add_argument('--rotate_degree', type=float, default=0)
        parser.add_argument('--mirror', type=str2bool, default=True)
        parser.add_argument('--is_pool', type=str2bool, default=True)


        parser.add_argument('--ids_per_batch', type=int, default=16)
        parser.add_argument('--ims_per_id', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=128)  # Testing Batch
        
        parser.add_argument('--log_to_file', type=str2bool, default=True)
        parser.add_argument('--steps_per_log', type=int, default=5)
        parser.add_argument('--epochs_per_val', type=int, default=1)
        parser.add_argument('--epochs_per_cluster', type=int, default=5)
        
        parser.add_argument('--last_conv_stride', type=int, default=1,
                            choices=[1, 2, 11])
        parser.add_argument('--normalize_feature', type=str2bool, default=False)
        parser.add_argument('--margin', type=float, default=0.5)
        parser.add_argument('--loss', type=str, default="Softmax")
        
        parser.add_argument('--only_test', type=str2bool, default=True)
        parser.add_argument('--resume', type=str2bool, default=True)
        parser.add_argument('--exp_dir', type=str, default='/home/dff/MFR/FaceRecognition/logs/mask')
        parser.add_argument('--model_weight_file', type=str, default='/ssd_datasets/dff/model/ckpt_best.pth')
        parser.add_argument('--model', type=str, default='Resnet50')
        
        parser.add_argument('--base_lr', type=float, default=0.01)#
        parser.add_argument('--lr_decay_type', type=str, default='warmup',
                            choices=['exp', 'staircase', 'warmup'])
        parser.add_argument('--exp_decay_at_epoch', type=int, default=61)  # 41
        parser.add_argument('--staircase_decay_at_epochs',
                            type=eval, default=(20, 60, 100))
        parser.add_argument('--staircase_decay_multiply_factor',
                            type=float, default=0.1) 
        parser.add_argument('--total_epochs', type=int, default=150)
        
        args = parser.parse_args()
        
        # gpu ids
        self.sys_device_ids = args.sys_device_ids
        
        # If you want to make your results exactly reproducible, you have
        # to fix a random seed.
        if args.set_seed:
            self.seed = 1
        else:
            self.seed = None
        
        if self.seed is not None:
            self.prefetch_threads = 1
        else:
            self.prefetch_threads = 1

        self.is_pool = args.is_pool
        
        # The experiments can be run for several times and performances be averaged.
        # `run` starts from `1`, not `0`.
        self.run = args.run
        
        ###########
        # Dataset #
        ###########
        self.partition_path = args.partition_path
        self.trainset_part = args.trainset_part
        
        # Image Processing
        self.crop_prob = args.crop_prob
        self.crop_ratio = args.crop_ratio
        self.resize_h_w = args.resize_h_w
        self.rotate_prob = args.rotate_prob
        self.rotate_degree=args.rotate_degree
        self.loss = args.loss
        # Whether to scale by 1/255
        self.scale_im = True
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]
        
        self.ids_per_batch = args.ids_per_batch
        self.ims_per_id = args.ims_per_id
        
        # training
        self.train_mirror_type = 'random' if args.mirror else None
        self.train_final_batch = False
        self.train_shuffle = True  # True
        
        self.test_batch_size = args.batch_size
        self.test_final_batch = True
        self.test_mirror_type = None
        self.test_shuffle = False
        
        dataset_kwargs = dict(
            resize_h_w=self.resize_h_w,
            scale=self.scale_im,
            im_mean=self.im_mean,
            im_std=self.im_std,
            batch_dims='NCHW',
            num_prefetch_threads=self.prefetch_threads)

        # train set
        prng = np.random
        if self.seed is not None:
            prng = np.random.RandomState(self.seed)
        self.train_set_kwargs = dict(
            path=self.partition_path,
            part=self.trainset_part,
            ids_per_batch=self.ids_per_batch,
            ims_per_id=self.ims_per_id,
            final_batch=self.train_final_batch,
            shuffle=self.train_shuffle,
            crop_prob=self.crop_prob,
            crop_ratio=self.crop_ratio,
            rotate_prob=self.rotate_prob,
            rotate_degree=self.rotate_degree,
            mirror_type=self.train_mirror_type,
            prng=prng)
        self.train_set_kwargs.update(dataset_kwargs)
        
        # test set
        prng = np.random
        if self.seed is not None:
            prng = np.random.RandomState(self.seed)
        self.test_set_kwargs = dict(
            path=self.partition_path,
            part='test',
            batch_size=self.test_batch_size,
            final_batch=self.test_final_batch,
            shuffle=self.test_shuffle,
            mirror_type=self.test_mirror_type,
            crop_prob=self.crop_prob,
            crop_ratio=self.crop_ratio,
            rotate_prob=0.0,
            prng=prng)
        self.test_set_kwargs.update(dataset_kwargs)

        
        ###############
        # ReID Model  #
        ###############
        # The last block of ResNet has stride 2. We can set the stride to 1 so that
        # the spatial resolution before global pooling is doubled.
        self.last_conv_stride = args.last_conv_stride
        
        # Whether to normalize feature to unit length along the Channel dimension,
        # before computing distance
        self.normalize_feature = args.normalize_feature
        
        # Margin of triplet loss(inter triplet, intra triplet)
        self.margin = args.margin
        self.model = args.model
        
        #############
        # Training  #
        #############
        self.weight_decay = 0.0005
        # Initial learning rate
        self.base_lr = args.base_lr
        self.lr_decay_type = args.lr_decay_type
        self.exp_decay_at_epoch = args.exp_decay_at_epoch
        self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
        self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
        # Number of epochs to train
        self.total_epochs = args.total_epochs
        
        # How often (in epochs) to test on val set.
        self.epochs_per_val = args.epochs_per_val
        self.epochs_per_cluster = args.epochs_per_cluster
        
        # How often (in batches) to log. If only need to log the average
        # information for each epoch, set this to a large value, e.g. 1e10.
        self.steps_per_log = args.steps_per_log
        self.only_test = args.only_test
        self.resume = args.resume
        
        #######
        # Log #
        #######
        
        # If True,
        # 1) stdout and stderr will be redirected to file,
        # 2) training loss etc will be written to tensorboard,
        # 3) checkpoint will be saved
        self.log_to_file = args.log_to_file
        
        # The root dir of logs.
        if args.exp_dir == '':
            self.exp_dir = osp.join(
                '{}'.format(self.model),
                '{}'.format(self.dataset),
                #
                'BR_lcs_{}_'.format(self.last_conv_stride) +
                'margin_{}_'.format(tfs(self.margin)) +
                'erasing_{}_'.format(self.rotate_prob) +
                '_{}_'.format(self.crop_prob) +
                'ids_{}_'.format(tfs(self.ids_per_batch)) +
                'ims_{}_'.format(tfs(self.ims_per_id)) +
                'lr_{}_'.format(tfs(self.base_lr,fmt='{:.7f}')) +
                '{}_'.format(self.lr_decay_type) +
                ('decay_at_{}_'.format(self.exp_decay_at_epoch)
                 if self.lr_decay_type == 'exp'
                 else 'decay_at_{}_factor_{}_'.format(
                    '_'.join([str(e) for e in args.staircase_decay_at_epochs]),
                    tfs(self.staircase_decay_multiply_factor))) +
                'total_{}'.format(self.total_epochs),
            )
        else:
            self.exp_dir = args.exp_dir
        
        self.stdout_file = osp.join(
            self.exp_dir, 'stdout_{}.txt'.format(time_str()))
        self.stderr_file = osp.join(
            self.exp_dir, 'stderr_{}.txt'.format(time_str()))
        
        # self.model.eval()
        # Saving model weights and optimizer states, for resuming.
        self.model_file=osp.join(
                '{}'.format(self.model))
        self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
        # Just for loading a pretrained model; no optimizer states is needed.
        self.model_weight_file = args.model_weight_file



def deep_clone_model(cfg, model, train_set_nums, TMO):

    state_dict = model.state_dict()
    model_copy = Model(last_conv_stride=cfg.last_conv_stride, num_classes=train_set_nums)
    model_copy.load_state_dict(state_dict)
    for param in model_copy.parameters():  # model.module.parameters
        param.requires_grad = False
    TMO([model_copy])
    return model_copy


def main():
    cfg = Config()
    
    # Redirect logs to both console and file.
    if cfg.log_to_file:
        ReDirectSTD(cfg.stdout_file, 'stdout', True)
        # ReDirectSTD(cfg.stderr_file, 'stderr', False)
     
    TVT, TMO = set_devices(cfg.sys_device_ids)
    
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    # Dump the configurations to log.
    import pprint
    print('-' * 60)
    print('cfg.__dict__')
    pprint.pprint(cfg.__dict__)
    print('-' * 60)
    
    ###########
    # Dataset #
    ###########
    test_set = create_dataset(**cfg.test_set_kwargs)

    
    ###########
    # Models  #
    ###########
    model = Model(last_conv_stride=cfg.last_conv_stride, num_classes = 10572,basenet=cfg.model, 
                  loss=cfg.loss, is_pool=cfg.is_pool)
    # model.load_state_dict(torch.load(osp.join(cfg.exp_dir, 'model.pth')))
    # Model wrapper
    # model_w = DataParallel(model)
    model = model.cuda()
    
    #############################
    # Criteria and Optimizers   #
    #############################
    
    base_params = list(model.base.parameters())  # finetuned
    new_params = [p for n, p in model.named_parameters()
                  if not n.startswith('base.')]
    param_groups = [{'params': base_params, 'lr': cfg.base_lr},
                    {'params': new_params, 'lr': cfg.base_lr}]

    optimizer = optim.SGD(param_groups,momentum = 0.9)    
    # Bind them together just to save some codes in the following usage.
    modules_optims = [model, optimizer]
    
    ################################
    # May Resume Models and Optims #
    ################################
    if cfg.resume:
        if cfg.model_weight_file != '':
            load_ckpt(modules_optims, cfg.model_weight_file)
        else:
            load_ckpt(modules_optims, cfg.ckpt_file)
    
    # May Transfer Models and Optims to Specified Device. Transferring optimizer
    # is to cope with the case when you load the checkpoint to a new device.
    TMO(modules_optims)


    def pre_process_im(img):
        """Pre-process image.
        `im` is a numpy array with shape [H, W, 3], e.g. the result of
        matplotlib.pyplot.imread(some_im_path), or
        numpy.asarray(PIL.Image.open(some_im_path))."""
        
        # Resize.
        if (cfg.resize_h_w is not None) \
            and (cfg.resize_h_w != (img.shape[0], img.shape[1])):
            
            img = cv2.resize(img, cfg.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)#INTER_CUBIC)
        
        # scaled by 1/255.
        if cfg.scale_im:
            img = img / 255.
        if cfg.im_mean is not None:
            img = img - np.array(cfg.im_mean)
        if cfg.im_mean is not None and cfg.im_std is not None:
            img = img / np.array(cfg.im_std).astype(float)

        # May mirror image.
        mirrored = False
        if cfg.test_mirror_type == 'always' \
            or (cfg.test_mirror_type == 'random' and cfg.prng.uniform() > 0.5):
            img = img[:, ::-1, :]
        mirrored = True

        # The original image has dims 'HWC', transform it to 'CHW'.
        img = img.transpose(2, 0, 1)

        return img, mirrored
    
    ###########
    # Testing #
    ###########


    def extract_feature(img):
        img = Variable(torch.from_numpy(img).float()).cuda()
        img = img.unsqueeze(0)
        feat,_, _, mask= model(img, [])
        feat = feat.data.cpu()
        _, mask = torch.max(mask.data.cpu(), 1)
        # print(mask)
        return feat, mask

    if cfg.only_test:
        if cfg.model_weight_file != '':
            map_location = (lambda storage, loc: storage)
            sd = torch.load(cfg.model_weight_file, map_location=map_location)
            load_state_dict(model, sd)
            print('Loaded model weights from {}'.format(cfg.model_weight_file))
        else:
            print(modules_optims)
            load_ckpt(modules_optims, cfg.ckpt_file)
        model.eval()
        print('Testing with model {}'.format(cfg.ckpt_file))

        video_names = ['DCM.ts', 'WMHD.ts', 'YTX.ts', 'HD01.mp4', 'HD02-1.mp4', 'HD02-2.mp4', 'BWBQ.ts']
        video_path = '/ssd_datasets/gmy/PKU-HumanID/'
        for v_name in video_names:
            v_path = os.path.join(video_path, v_name)
            pkl_path = os.path.join(video_path, v_name.split('.')[0]+'.pkl')
            print(v_path)
            print(pkl_path)
            info_dict = load_pkl(pkl_path)  
            print(len(info_dict['detected_faces']))

            for idx in range(len(info_dict['detected_faces'])):
                img_info = info_dict['detected_faces'][idx]
                if 'masked_face' in img_info.keys():
                    img = img_info['masked_face']
                    info_dict['detected_faces'][idx]['mask_label'] = 1          
                else:
                    img = img_info['aligned_face']
                    info_dict['detected_faces'][idx]['mask_label'] = 0          
                img = np.asarray(img)
                # print(img)
                img, _ = pre_process_im(img)
                # img = torch.from_numpy(img)
                # print(img)
                feat, mask = extract_feature(img)
                feat = feat.numpy()[0].tolist()
                mask = mask.numpy()[0]
                
                info_dict['detected_faces'][idx]['feat'] = feat
                info_dict['detected_faces'][idx]['mask'] = mask
                
                # print(feat)
                # print(len(feat))
                # print(mask)


            output_file = os.path.join(video_path, v_name.split('.')[0]+'_feat.pkl')
            with open(output_file, 'wb') as outp:  # Overwrites any existing file.
                 pickle.dump(info_dict, outp, pickle.HIGHEST_PROTOCOL)
        return
    

if __name__ == '__main__':
    main()


import numpy as np
import os.path as osp
ospj = osp.join
ospeu = osp.expanduser

from ..utils.utils import load_pickle
from ..utils.dataset_utils import parse_im_name
from .TrainSet import TrainSet
from .TrainSetLandmark import TrainSetLandmark
from .TestSet import TestSet


def create_dataset(
    path=None,
    part='all',
    **kwargs):
  ########################################
  # Specify Directory and Partition File #
  ########################################
  im_type='p'

  partition_file = ospeu(path)
  ##################
  # Create Dataset #
  ##################

  # Use standard Market1501 CMC settings for all datasets here.
  cmc_kwargs = dict(separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
  
  if part == "test_single":
      marks = [int(0)]
      im_names = [path]
      im_ids = [int(0)]
      im_cams = [int(0)]

      kwargs.update(cmc_kwargs)
      ret_set = TestSet(
        im_names=im_names,
        im_ids = im_ids,
        im_cams = im_cams,
        marks=marks,
        im_type=im_type,
        **kwargs)
      return ret_set
  
  partitions = load_pickle(partition_file)
  #im_names = partitions['{}_im_names'.format(part)]
  
  if part == 'trainval_landmark':
    ids2labels = partitions['trainval_ids2labels']
    im_names = partitions['trainval_im_names']
    im_ids = partitions['trainval_im_ids']
    scales = partitions['trainval_im_scales']
    # scales = []
    ret_set = TrainSetLandmark(
      im_names=im_names,
      im_scales=scales,
      ids2labels=ids2labels,
      im_ids = im_ids,
      im_type=im_type,
      **kwargs)

  elif part == 'trainval':
    ids2labels = partitions['trainval_ids2labels']
    im_names = partitions['trainval_im_names']
    im_ids = partitions['trainval_im_ids']
    ret_set = TrainSet(
      im_names=im_names,
      ids2labels=ids2labels,
      im_ids = im_ids,
      im_type=im_type,
      **kwargs)

  elif part == 'test':
    if im_type=='p':
      marks = partitions['test_marks']
      im_names = partitions['test_im_names']
      im_ids = partitions['test_ids']
      im_cams = partitions['test_cams']
    else:
      marks=None

    kwargs.update(cmc_kwargs)
    ret_set = TestSet(
      im_names=im_names,
      im_ids = im_ids,
      im_cams = im_cams,
      marks=marks,
      im_type=im_type,
      **kwargs)

  return ret_set

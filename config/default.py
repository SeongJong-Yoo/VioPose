# Copyright (c) Microsoft
# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Seong Jong Yoo (yoosj@umd.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = 'Logs/test1'
_C.PRINT_FREQ = 20
_C.EPOCH = 100
_C.BATCH_SIZE = 64
_C.SAVE_PERIOD = 10
_C.MODE = 'train'
_C.CONTINUE = False

# Dataset Parameters
_C.DATA = CN()
_C.DATA.DIR = ''
_C.DATA.SPLIT_RATIO = 0.1       # validation: 0.1 and train: 0.9
_C.DATA.AUDIO_TYPE = 'feature'  #'raw' or 'features' or 'high_features' or 'spectrogram' or 'spectrogram_100FPS'
_C.DATA.MFCC_ONLY = False
_C.DATA.FILTERED = False
_C.DATA.POSE_TYPE='None'    # None or gt or openpose
_C.DATA.STRIDE = 30
_C.DATA.PAD = 0


# Optimizer Parameters
_C.OPT = CN()
_C.OPT.TYPE = 'Adam'
_C.OPT.SCHEDULE = 'ExponentialDecay'    # 'ExponentialDecay' or 'PiecewiseConstantDecay'
_C.OPT.DECAY_RATE = 0.95
_C.OPT.DECAY_STEPS = 185        # Depends on bath size and data size (# steps = 1 epoch)
_C.OPT.INIT_LR = 1e-4
_C.OPT.BOUNDARIES=[10000, 20000]        # Initial 10k steps 10k ~ 20k and the others
_C.OPT.LR_VALUES=[0.01, 0.001, 0.0001]     
_C.OPT.STAIRCASE = True
_C.OPT.FULL_WEIGHT = 0.5
_C.OPT.CENTER_WEIGHT = 0.5
_C.OPT.METRIC_TYPE = ["loss", "mpjpe", "mpjae"]

# Network Parameters
_C.MODEL = CN()
_C.MODEL.NAME = 'MAPNET Test'
_C.MODEL.TYPE = 'baseline'
_C.MODEL.CENTER_OUTPUT = False
_C.MODEL.NUM_KEYPOINTS = 12
_C.MODEL.HIDDEN_DIM_RATIO = 2
_C.MODEL.NUM_HEADS = 8
_C.MODEL.DROPOUT = 0.1
_C.MODEL.OUTPUT_FRAME = 90
_C.MODEL.AUDIO_ERROR = 'vel' # pose or vel
_C.MODEL.VIDEO_FPS = 30

# Pose Module
_C.MODEL.POSE_MODULE = CN()
_C.MODEL.POSE_MODULE.LAYERS = 3
_C.MODEL.POSE_MODULE.CROSS_LAYERS = 3
_C.MODEL.POSE_MODULE.DIM = 16
_C.MODEL.POSE_MODULE.FRAME = 90
_C.MODEL.POSE_MODULE.TYPE = 'spatial'     # spatial or temporal

# Dynamic Module
_C.MODEL.DYNAMIC_MODULE = CN()
_C.MODEL.DYNAMIC_MODULE.DIM = 256
_C.MODEL.DYNAMIC_MODULE.HIDDEN_DIM_RATIO=2

# Audio Module
_C.MODEL.AUDIO_MODULE = CN()
_C.MODEL.AUDIO_MODULE.REPRESENTATION_TYPE = '6D'               # '6D': 6D rotation representation
_C.MODEL.AUDIO_MODULE.FRAME = 24000                            # Raw audio is sampled 8k SR and 3 second length
_C.MODEL.AUDIO_MODULE.CNN_TYPE='1D'                            # 1D or 2D (2D is only for features)
_C.MODEL.AUDIO_MODULE.CNN_FILTERS = [256, 256, 256, 128, 64]   # 1D CNN filters
_C.MODEL.AUDIO_MODULE.CNN_KERNELS = [5, 5, 5, 5, 5]            # 1D CNN kernel size
_C.MODEL.AUDIO_MODULE.CNN_STRIDES = [1, 1, 1, 1, 1]            # 1D CNN stride
_C.MODEL.AUDIO_MODULE.TRANS_LAYERS = 3                         # Number of Transformer layers
_C.MODEL.AUDIO_MODULE.CROSS_LAYERS = 3                         # Number of Cross-Transformer layers
_C.MODEL.AUDIO_MODULE.DIM = 128

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.MODEL_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA.DIR = args.dataDir

    if args.mode:
        cfg.MODE = args.mode
        
    cfg.freeze()

def read_config(cfg, file):
    cfg.defrost()
    cfg.merge_from_file(file)
    cfg.freeze()
import argparse
import numpy as np
import os
from tqdm import tqdm

from utility.loss import *
from utility.data_loader import * 
from utility.utils import *
from utility.logger import *
from config import cfg
from config import read_config
from main import build_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_args():
    parser = argparse.ArgumentParser(description='MAPNet Argument Parser')
    parser.add_argument('--folder',
                        help='Experiment folder name',
                        required=True,
                        type=str)
    parser.add_argument('--type',
                        default='best',
                        help='Which type model will you use',
                        metavar="<best|last>")
    parser.add_argument('--save',
                        default=True,
                        help='Save the result')
    parser.add_argument('--data',
                        default='violin',
                        help='dataset type',
                        metavar="<violin/video/other>",
                        type=str)
    parser.add_argument('--data_path',
                        default='violin',
                        help='/DATA/PATH',
                        type=str)
    args = parser.parse_args()

    return args

envelope = {'gt_poses': [],
            'pred_poses': [],
            'input_2d': [],
            'info_list': [],
            'gt_classes': [],
            'pred_classes': [],
            'att_weights': []}

def merge_upsampling(envelope, overlapped_frame):
    pose_up = {}
    pose_up_gt = {}
    len_trackers = {}

    for i, item in enumerate(envelope['info_list']):
        value_pred_up = envelope['pred_poses_up'][i].numpy()

        for j, key in enumerate(item['trial']):
            key = key.numpy().decode('ascii')
            if key not in pose_up:
                pose_up[key] = []
                pose_up_gt[key] = []
                len_trackers[key] = 0

            start = item['start'][j].numpy()

            start_frame = int((start / item['fps'][j].numpy()) * 100)
            end_frame = start_frame + len(value_pred_up[j])
            if end_frame > len(dataset['kp_optim_3d'][key]):
                leftover = len(dataset['kp_optim_3d'][key]) - start_frame
                pad_left_size = len(value_pred_up[j]) - leftover
                end_frame = len(dataset['kp_optim_3d'][key])
            else:
                pad_left_size = 0


            pose_up[key].append(value_pred_up[j][pad_left_size:])
            pose_up_gt[key].append(dataset['kp_optim_3d'][key][start_frame:end_frame])
            len_trackers[key] += len(value_pred_up[j][pad_left_size:])

    out_pred_up = {}
    out_pred_up_gt = {}
    for key in pose_up.keys():
        if overlapped_frame==0:
            frame_compensate = int(-len(pose_up[key][0]))
        else:
            frame_compensate = int(frame_compensate * 100 / item['fps'][0])
        out_pred_up[key] = pose_up[key][0][:-frame_compensate]
        out_pred_up_gt[key] = pose_up_gt[key][0][:-frame_compensate]

        for i in range(len(pose_up[key]) - 1):
            if i==len(pose_up[key]) - 2:
                out_pred_up[key] = np.concatenate([out_pred_up[key], pose_up[key][i+1]], axis=0) 
                out_pred_up_gt[key] = np.concatenate([out_pred_up_gt[key], pose_up_gt[key][i+1]], axis=0) 
            else:
                out_pred_up[key] = np.concatenate([out_pred_up[key], pose_up[key][i+1][:-frame_compensate]], axis=0) 
                out_pred_up_gt[key] = np.concatenate([out_pred_up_gt[key], pose_up_gt[key][i+1][:-frame_compensate]], axis=0) 
    
    return out_pred_up, out_pred_up_gt

def merge(envelope, overlapped_frame):
    audio = False
    if len(envelope['out_audio_pose']) !=0:
        audio = True

    output = {'gt': {},
              'gt_2d': {},
              'pred': {},
              'input': {}}
    if audio:
        output['out_audio_pose'] = {}
    if KALMAN_FILTER:
        output['acc'] = {}
        output['vel'] = {}

    for i, item in enumerate(envelope['info_list']):
        for key in output.keys():
            value = envelope[key][i].numpy()


            for j, trial in enumerate(item['trial']):
                start = item['start'][j].numpy()
                end = item['end'][j].numpy()
                L = end - start
                offset = int((L - overlapped_frame) / 2)
                trial = trial.numpy().decode('ascii')
                
                end = end - offset
                if end > len(dataset['kp_optim_3d_vid'][trial])-1:
                    end = len(dataset['kp_optim_3d_vid'][trial]) - start
                else:
                    end = offset + overlapped_frame
                start = offset
                if trial not in output[key]:
                    output[key][trial] = value[j][start:end]
                else:
                    output[key][trial] = np.concatenate((output[key][trial], value[j][start:end]), axis=0)

    for key in output.keys():
        for trial in output[key].keys():
            if len(output[key][trial]) != len(dataset['kp_optim_3d_vid'][trial]):
                print("Error: Length is not correct")
                exit()

    return output

def record_result(result, gt, input, output, cfg):
    pose_x = input['pose']
    num_frame = output.shape[1]

    if cfg.NUM_KEYPOINTS==8:
        pose_y = gt['pose']
        pose_x = pose_x
    elif cfg.NUM_KEYPOINTS==12:
        pose_y = gt['pose'][:, :, :12, :]
    else:
        pose_y = gt['pose']

    if num_frame==90:
        pose_y = gt['pose_2d']

    result['gt']['pose'] = pose_y
    result['pred']['pose'] = output
    
    return result

def evaluate(single_dataset, overlapped_frame, att=False):
    envelope = {'gt': [],
                'gt_2d': [],
                'pred': [],
                'input': [],
                'out_audio_pose': [],
                'info_list': []}

    if KALMAN_FILTER:
        envelope['acc'] = []
        envelope['vel'] = []

    for input, gt, info in tqdm(single_dataset):
        pose_x = input['pose']
        if 'audio' in input:
            audio_x = input['audio']
        else:
            audio_x = None

        output = model([pose_x, audio_x], training=False)

        envelope['gt'].append(gt['pose'])
        envelope['pred'].append(output['spatial'][1])
        envelope['input'].append(input['pose'])
        envelope['gt_2d'].append(gt['pose_2d'])
        envelope['info_list'].append(info)
        if 'audio_joints' in output:
            envelope['out_audio_pose'].append(output['audio_joints'])
        if att:
            np.savez_compressed('./att_weights', **output['att_weights'])
            np.savez_compressed('./att_weights_others', **envelope)
            att = False
        if 'spatial_up' in output:
            envelope['pred_up'].append(output['spatial_up'])
        if KALMAN_FILTER:
            envelope['acc'].append(output['acc'])
            envelope['vel'].append(output['vel'])

    return merge(envelope, overlapped_frame)


overlapped_frame = cfg.MODEL.OUTPUT_FRAME
# overlapped_frame = 30
KALMAN_FILTER=True
args = parse_args()
cfg_path = os.path.join(args.folder, 'cfg_parameters.yaml')
read_config(cfg, cfg_path)
cfg.defrost()
cfg.MODE='test'
cfg.DATA.FILTERED=False
cfg.DATA.DIR=args.data_path
cfg.DATA.STRIDE=overlapped_frame
cfg.freeze()

dataset=ViolinDataset(cfg, train=False, data_type=args.data)

_, test_ds, audio_frame, audio_dim = slice_generate_dataset(dataset, cfg, test=True)

# Load Model
model, ONLY_LOSS = build_model(cfg, audio_frame, audio_dim)
model = read_model(args.folder, args.type, model)

result = {}

output = evaluate(test_ds, overlapped_frame, False)

if KALMAN_FILTER:
    for key in output['pred'].keys():
        output['pred'][key]=kalman_filter(output['pred'][key], output['vel'][key], output['acc'][key])

result['gt']=output['gt']
result['gt_2d']=output['gt_2d']
result['pred']=output['pred']
result['input']=output['input']
if 'out_audio_pose' in output:
    result['out_audio_pose']=output['out_audio_pose']

if 'pred_up' in output:
    result['pred_up']=output['pred_up']
    result['pred_up_gt']=output['pred_up_gt']


print("Saving the result")
result_path = os.path.join(args.folder, 'result')
np.savez_compressed(result_path, **result)

if args.data=='violin':
    print("Compute and save evaluation indices")
    compute_values(result, args.folder)


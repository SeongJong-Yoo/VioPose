import tensorflow as tf
import numpy as np
import sys
from utility.loss import *
from termcolor import colored
from einops.layers.tensorflow import Rearrange
import os
import librosa

def pick_upper_body(input, neck_centered=False):
    '''
    Pick only upper body joints
            body joints(batch, frame, 14, dim)
        Input:
                Left:  0: Toe, 3: Knee, 2: Pelvis, 4: Shoulder,  1: Elbow, 5: Wrist, 12: Hand
                Right: 6: Toe, 9: Knee, 8: Pelvis, 10: Shoulder, 7: Elbow, 11: Wrist, 13: Hand
            Body Joints: 17 joints (batch, 17, 3)
                Left:  0: Toe, 3: Knee, 2: Pelvis, 4: Shoulder,  1: Elbow, 5: Wrist, 12: LBackHand, 13: RBackHand
                Right: 6: Toe, 9: Knee, 8: Pelvis, 10: Shoulder, 7: Elbow, 11: Wrist, 15: LBackHand, 16: RBackHand
        Output: 
            8 joints (batch, 8, dim)
                Left:  0: Shoulder,  1: Elbow, 2: Wrist, 3: Hand
                Right: 4: Shoulder, 5: Elbow, 6: Wrist, 7: Hand
    '''
    shape = input.shape
    if len(shape) == 4:
        input = tf.transpose(input, [2, 0, 1, 3])  # (14, batch, frame, dim)
        num_joint = shape[2]
    elif len(shape) == 5:
        input = tf.transpose(input, [2, 0, 1, 3, 4])  # (14, batch, frame, dim, dim)
        num_joint = shape[2]
    elif len(shape)==3:
        input = tf.transpose(input, [1, 0, 2])
        num_joint = shape[1]
    
    if num_joint == 14:
        output = tf.stack([
            input[4],
            input[1],
            input[5],
            input[12],
            input[10],
            input[7],
            input[11],
            input[13],
        ], axis=0)
    elif num_joint == 17:
        output = tf.stack([
            input[4],
            input[1],
            input[5],
            (input[12] + input[13]) / 2,
            input[10],
            input[7],
            input[11],
            (input[15] + input[16]) / 2,
        ], axis=0)
    else:
        print("Error: This function only gets 14 or 17 joints")
        exit()
        
    if len(shape) == 4:
        output = tf.transpose(output, [1, 2, 0, 3])
        if neck_centered:
            neck = (input[4] + input[10])/2
            return output - neck[:, :, tf.newaxis, :]
    elif len(shape) == 5:
        output = tf.transpose(output, [1, 2, 0, 3, 4])
        if neck_centered:
            neck = (input[4] + input[10])/2
            return output - neck[:, :, tf.newaxis, :]
    elif len(shape)==3:
        output = tf.transpose(output, [1, 0, 2])
        if neck_centered:
            neck = (input[4] + input[10])/2
            return output - neck[:, tf.newaxis, :]

    return output

def continue_setting(cfg, model=None, logger=None):
    model_dir = os.path.join(cfg.LOG_DIR, "model")
    is_continue = cfg.CONTINUE

    if is_continue == True and os.path.exists(model_dir) == False:
        is_continue = False
        print("THERE IS NO SAVED MODEL. START FROM THE BEGINNING")
        start_epoch = 1

    if is_continue == True:
        key = 'last'
        model_dir = [f for f in os.listdir(os.path.join(cfg.LOG_DIR, 'model'))]
        model_dir = [i for i in model_dir if key in i]
        metric_dir = [f for f in os.listdir(os.path.join(cfg.LOG_DIR, 'metric'))]
        metric_dir = [i for i in metric_dir if key in i]

        if len(model_dir) == 0 or len(metric_dir) == 0:
            is_continue = False
            print("THERE IS NO SAVED MODEL. START FROM THE BEGINNING")

        else:
            start_epoch = int(model_dir[-1].split('_')[2].split('.')[0])
            start_epoch = start_epoch + 1
            model_dir = os.path.join(cfg.LOG_DIR, 'model', model_dir[-1])
            model.load_weights(model_dir)

            metric_dir = os.path.join(cfg.LOG_DIR, 'metric', metric_dir[-1])
            history = np.load(metric_dir, allow_pickle=True)[()]
            logger.update_logger(history, model_dir, metric_dir)
            logger.make_tensorboard(cfg.LOG_DIR)

    else:
        logger.save_cfg(cfg)
        logger.make_tensorboard(cfg.LOG_DIR)
        start_epoch = 1

    return model, logger, start_epoch

def read_model(dir, type, model):
    model_dir = [f for f in os.listdir(os.path.join(dir, "model"))]
    model_dir = [i for i in model_dir if type in i]

    if len(model_dir) == 0 :
        print("THERE IS NO SAVED MODEL. ")
        exit()
    
    model_dir = os.path.join(dir, 'model', model_dir[-1])
    model.load_weights(model_dir, by_name=True)
    print("Successfully load pre-trained model")

    return model

def reporter(metric_obj, epoch, train_time, val_time, ONLY_LOSS):
    borderline = '=================================================================================================='
    train_time = "{:.4f}".format(train_time/60)
    val_time = "{:.4}".format(val_time/60)

    print(colored(borderline, 'red', 'on_white'))
    print(colored('Epoch: ', 'red'), epoch, '\t', colored('Train Time(min): ', 'blue'), train_time, '\t', colored('Val Time(min): ', 'green'), val_time)
    if ONLY_LOSS:
        template = 'Train Loss: {:.4f}\t Val Loss: {:.4f}\t \n'
        print(template.format(metric_obj['train_loss'].result().numpy(),
                            metric_obj['val_loss'].result().numpy()))
    else:
        template = 'Train Loss: {:.4f}\t Train MPJPE: {:.4f} \t Train MPJAE: {:.4f} \n'+\
                'Val Loss: {:.4f}\t Val MPJPE: {:.4f} \t Val MPJAE: {:.4f} \n'
        print(template.format(metric_obj['train_loss'].result().numpy(),
                            metric_obj['train_mpjpe'].result().numpy(),
                            metric_obj['train_mpjae'].result().numpy(),
                            metric_obj['val_loss'].result().numpy(),
                            metric_obj['val_mpjpe'].result().numpy(),
                            metric_obj['val_mpjae'].result().numpy()))

def report_all(metric_obj):
    for key, value in metric_obj.items():
        template = key + ' {:.4f}\n'
        print(template.format(metric_obj[key].result().numpy()))

def save_result(root, result, metric):
    result_path = os.path.join(root, 'result')
    np.save(result_path, result)
    metric_path = os.path.join(root, 'metric.txt')

    with open(metric_path, 'w') as f:
        for key, value in metric.items():
            line = key + ': ' + str(value.result().numpy()) + ' \n'
            f.write(line)

def compute_values(data, path):
    metric = {'mpjpe': [],
            'mpjae': [],
            'mpjve': [],
            'pa_mpjpe': [],
            'mpjpe_joint': []}
    
    if 'gt' in data:
        for key in data['gt'].keys():
            gt = align_by_center(np.expand_dims(np.asarray(data['gt'][key], dtype=np.float32), axis=0))
            pred = align_by_center(np.expand_dims(np.asarray(data['pred'][key]), axis=0))

            metric['mpjpe'].append(mpjpe(gt, pred))
            metric['mpjae'].append(mpjae(gt, pred))
            metric['pa_mpjpe'].append(pa_mpjpe(gt, pred))
            metric['mpjpe_joint'].append(mpjpe_jointwise(gt, pred))
            metric['mpjve'].append(mpjve(gt, pred))

    metric_path = os.path.join(path, 'index.txt')

    with open(metric_path, 'w') as f:
        for key in metric.keys():
            if len(metric[key]) > 0:
                value = np.mean(metric[key], axis=0)
                if key=='mpjpe_joint':
                    for i in range(len(value)):
                        line = 'joint_' + str(i) + ": " + str(value[i]) + '\n'
                        f.write(line)
                else:
                    line = key + ": " + str(value) + '\n'
                    f.write(line)


def load_optimizer(cfg):
    if cfg.SCHEDULE == 'ExponentialDecay':
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=cfg.INIT_LR,
            decay_steps=cfg.DECAY_STEPS,
            decay_rate=cfg.DECAY_RATE,
            staircase=cfg.STAIRCASE
        )
    elif cfg.SCHEDULE=='PiecewiseConstantDecay':
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=cfg.BOUNDARIES,
            values=cfg.LR_VALUES
        )
    elif cfg.SCHEDULE == None:
        lr_schedule = cfg.INIT_LR


    if cfg.TYPE == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=True)

    elif cfg.TYPE == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=cfg.MOMENTUM)
    return opt

def get_metrics(metric_type, test=False, total=True, joint_wise=False):
    metric_obj = dict()
    if test:
        for type in metric_type:
            if type=='Acc':
                metric_obj["test_" + type] = tf.keras.metrics.CategoricalAccuracy()
            else:
                metric_obj["test_" + type] = tf.keras.metrics.Mean()
        if total==False:
            metric_obj["test_init"] = tf.keras.metrics.Mean() 
            metric_obj["test_mid"] = tf.keras.metrics.Mean() 
            metric_obj["test_final"] = tf.keras.metrics.Mean() 
        if joint_wise:
            for i in range(8):
                key = "test_joint" + str(i)
                metric_obj[key] = tf.keras.metrics.Mean()
        metric_obj['test_pa-mpjpe']=tf.keras.metrics.Mean()
    else:
        for type in metric_type:
            if type=='Acc':
                metric_obj["train_" + type] = tf.keras.metrics.CategoricalAccuracy()
                metric_obj["val_" + type] = tf.keras.metrics.CategoricalAccuracy()
            else:
                metric_obj["train_" + type] = tf.keras.metrics.Mean()
                metric_obj["val_" + type] = tf.keras.metrics.Mean()
    
    return metric_obj

def metric_reset(metric_obj):
    for _, metric in metric_obj.items():
        metric.reset_states()

def loss_compute(gt, output, cfg, audio, total=True, joint_wise=False):
    loss_obj={}

    pose_y = gt['pose']

    pose_x = output['spatial'][1]
    pred_acc = output['acc']
    pred_vel = output['vel']

    gt_vel = pose_y[:, 1:] - pose_y[:, :-1]
    gt_vel = tf.concat((gt_vel, gt_vel[:, -1:, :]), axis=1)
    gt_acc = pose_y[:, 2:] - 2 * pose_y[:, 1:-1] + pose_y[:, :-2]
    gt_acc = tf.concat((gt_acc, gt_acc[:, -2:, :]), axis=1)

    if cfg.DATA.FILTERED:
        frame = gt_vel.shape[1] / 3
        b, a = signal.butter(2, int(frame/4), fs=frame, btype='low', analog=False)
        b = tf.constant(b, dtype=tf.float32)
        a = tf.constant(a, dtype=tf.float32)
        gt_vel = apply_filter(gt_vel, b, a)
        gt_acc = apply_filter(gt_acc, b, a)

    loss_obj['mpjpe'] = mpjpe(pose_y, pose_x)
    loss_obj['mpjve'] = mpjve(pose_y, pose_x)
    loss_obj['mpjae'] = mpjae(pose_y, pose_x)
    loss_obj['sim'] = normal_sim(pose_y, pose_x)
    loss_obj['CosAcel'] = max_cos_sim(gt_acc, pred_acc)
    loss_obj['CosVel'] = max_cos_sim(gt_vel, pred_vel)

    if cfg.MODE=='test':
        loss_obj['pa-mpjpe'] = pa_mpjpe(pose_y, pose_x)

    if 'error' in output:
        loss_obj['error'] = tf.math.reduce_mean(output['error'])

    loss_obj['loss'] = loss_obj['mpjpe'] + 100 * loss_obj['CosAcel'] + 100 * loss_obj['CosVel']

    if total==False:
        loss_obj['init'] = mpjpe(pose_y[:, :100, :, :], pose_x[:, :100, :, :])
        loss_obj['mid'] = mpjpe(pose_y[:, 100:200, :, :], pose_x[:, 100:200, :, :])
        loss_obj['final'] = mpjpe(pose_y[:, 200:300, :, :], pose_x[:, 200:300, :, :])
    if joint_wise:
        joint_mpjpe = mpjpe_jointwise(pose_y, pose_x)
        for i in range(len(joint_mpjpe)):
            key = 'joint' + str(i)
            loss_obj[key] = joint_mpjpe[i]
    return loss_obj


def metric_update(metric_obj, loss_obj, mode='train'):

    for _, key in enumerate(metric_obj):
        label = key.split('_')
        if label[0] == mode:
            if label[1]=='accuracy':
                metric_obj[key].update_state(loss_obj['gt'], loss_obj['pred'])
            elif 'Acc' in label[1]:
                metric_obj[key].update_state(loss_obj['gt_class'], loss_obj['pred_class'])
            else:
                metric_obj[key].update_state(loss_obj[label[1]])


def kalman_filter(pose, vel, acc, fs=30):
    """ 
    Kalman Filter
    State Variable x: (p, dp/dt, d^2p/dt^2)
    """
    f, j, _ = pose.shape
    x = np.concatenate((pose, vel, acc), axis=-1)
    # State transition matrix A
    dt = 30 / fs
    A = np.array([
        [1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
        [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
        [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
        [0, 0, 0, 1, 0, 0, dt, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, dt, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, dt],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])

    ## 0: Left Ankle, 1: Left Elbow, 2: Left Pelvis, 3: Left Knee, 4: Left Shoulder, 5: Left Wrist
    ## 6: Right Ankle, 7: Right Elbow, 8: Right Pelvis, 9: Right Knee, 10: Right Shoulder, 11: Right Wrist
    ## 12: Left Index, 13: Left Pinky, 14: Left Thumb
    ## 15: Right Index, 16: Right Pinky, 17: Right Thumb

    cov = np.array([10, 1, 0.1, 1, 0.1, 10, 10, 1, 0.1, 1, 0.1, 10, 10, 10, 10, 10, 10, 10])
    # Process noise covariance Q
    Q_format = np.array([
        [0.25*dt**4, 0.25*dt**4, 0.25*dt**4, 0.5*dt**3, 0.5*dt**3, 0.5*dt**3, 0.5*dt**2, 0.5*dt**2, 0.5*dt**2],
        [0.25*dt**4, 0.25*dt**4, 0.25*dt**4, 0.5*dt**3, 0.5*dt**3, 0.5*dt**3, 0.5*dt**2, 0.5*dt**2, 0.5*dt**2],
        [0.25*dt**4, 0.25*dt**4, 0.25*dt**4, 0.5*dt**3, 0.5*dt**3, 0.5*dt**3, 0.5*dt**2, 0.5*dt**2, 0.5*dt**2],
        [0.5*dt**3, 0.5*dt**3, 0.5*dt**3, dt**2, dt**2, dt**2, dt, dt, dt],
        [0.5*dt**3, 0.5*dt**3, 0.5*dt**3, dt**2, dt**2, dt**2, dt, dt, dt],
        [0.5*dt**3, 0.5*dt**3, 0.5*dt**3, dt**2, dt**2, dt**2, dt, dt, dt],
        [0.5*dt**2, 0.5*dt**2, 0.5*dt**2, dt, dt, dt, 1, 1, 1],
        [0.5*dt**2, 0.5*dt**2, 0.5*dt**2, dt, dt, dt, 1, 1, 1],
        [0.5*dt**2, 0.5*dt**2, 0.5*dt**2, dt, dt, dt, 1, 1, 1]
    ])
    Q_eye = np.eye(9)
    Q = np.stack([Q_eye * cov[i] for i in range(j)], axis=0)

    # Measurement noise covariance R
    R_eye = np.eye(9)
    R = np.stack([R_eye * cov[i] for i in range(j)], axis=0) 

    # Initial estimate covariance P
    P0 = np.tile(np.eye(9), (j, 1, 1))
    x_estimates = [x[0]]
    P_estimates = [P0]

    for i in range(1,f):
        # Prediction step
        x_pred = np.einsum('ij, fj->fi', A, x_estimates[-1])
        P_pred = np.einsum('fij, kj->fik', P_estimates[-1], A)
        P_pred = np.einsum('ji, fik->fjk', A, P_pred) + Q

        # Update step
        y = x[i] - x_pred
        S = P_pred + R
        K = P_pred @ np.linalg.inv(S)

        x_est = x_pred + np.einsum('fij, fj->fi', K, y)
        P_est = (np.tile(np.eye(9), (j, 1, 1)) - K) @ P_pred

        x_estimates.append(x_est)
        P_estimates.append(P_est)

    return np.array(x_estimates)[:, :, :3]

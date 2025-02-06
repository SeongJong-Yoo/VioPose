import argparse
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import sys
from einops.layers.tensorflow import Rearrange

# import _init_paths
from utility.loss import *
from utility.data_loader import * 
from utility.utils import *
from utility.logger import *
from config import cfg
from config import update_config
from model import VioPose


def parse_args():
    parser = argparse.ArgumentParser(description='MAPNet Argument Parser')
    parser.add_argument('--cfg',
                        help='Experiment Configuration File Name',
                        required=True,
                        type=str)
    parser.add_argument('--mode',
                        default="train",
                        required=False,
                        metavar="<train|test>")
    
    parser.add_argument('--modelDir',
                        help='Previous Model Directory',
                        required=False,
                        type=str)
    parser.add_argument('--logDir',
                        help='Log Directory',
                        required=False,
                        type=str)
    parser.add_argument('--dataDir',
                        help='Dataset Directory',
                        required=False,
                        type=str)
  
    args = parser.parse_args()

    return args

def build_model(cfg, audio_frame, audio_dim):
    # Build model
    ONLY_LOSS = False
    if audio_frame is None:
        audio_frame = 0
        audio_dim = 0

    pos_x_dim = np.asarray([cfg.BATCH_SIZE, cfg.MODEL.POSE_MODULE.FRAME, cfg.MODEL.NUM_KEYPOINTS, 2])
    audio_x_dim = np.asarray([cfg.BATCH_SIZE, audio_frame, audio_dim])
    if cfg.DATA.AUDIO_TYPE=='raw':
        audio_x_dim = np.asarray([cfg.BATCH_SIZE, audio_frame])
    
    if cfg.MODEL.TYPE=='viopose':
        model = VioPose.construct_model(cfg.MODEL, cfg.DATA.AUDIO_TYPE, pos_x_dim, audio_x_dim)
    else:
        print("Model not found")
        sys.exit()

    return model, ONLY_LOSS

def main():
    ONLY_LOSS=False
    args = parse_args()
    update_config(cfg, args)

    # GPU setup
    gpus = tf.config.list_logical_devices('GPU')
    print("Number of GPUS: ", len(gpus))
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy(gpus)
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Load dataset
    dataset = ViolinDataset(cfg, train=True)

    train_ds, val_ds, audio_frame, audio_dim = slice_generate_dataset(dataset, cfg)

    if len(gpus) > 1:
        train_ds = strategy.experimental_distribute_dataset(train_ds)
        val_ds = strategy.experimental_distribute_dataset(val_ds)
        
    # Parameters for Log (Model, Metrics, Tensorboard)
    logger = Logger(cfg.OPT.METRIC_TYPE)
    if not os.path.exists('./Logs'):
        os.mkdir('./Logs')
    if not os.path.exists(cfg.LOG_DIR):
        os.mkdir(cfg.LOG_DIR)

    logger.save_cfg(cfg)
    logger.make_tensorboard(cfg.LOG_DIR)
    
    if len(gpus) > 1:
        with strategy.scope():
            model, ONLY_LOSS = build_model(cfg, audio_frame, audio_dim)
            model, logger, start_epoch = continue_setting(cfg, model, logger)
            # Setup training parameters
            optimizer = load_optimizer(cfg.OPT)
            metric_obj = get_metrics(cfg.OPT.METRIC_TYPE)
    else:
        model, ONLY_LOSS = build_model(cfg, audio_frame, audio_dim)
        model, logger, start_epoch = continue_setting(cfg, model, logger)
        # Setup training parameters
        optimizer = load_optimizer(cfg.OPT)
        metric_obj = get_metrics(cfg.OPT.METRIC_TYPE)

    @tf.function
    def step(input, gt, split):
        training=True
        if split=='val':
            training = False
        pose_x = input['pose']
        if 'audio' in input:
            audio_x = input['audio']
        else:
            audio_x = np.zeros_like(pose_x)

        with tf.GradientTape() as tape:
            output = model([pose_x, audio_x], training=training)

            loss_obj = loss_compute(gt, output, cfg, audio=audio_x)
            metric_update(metric_obj, loss_obj, mode=split)
            if split=='train':
                gradients = tape.gradient(loss_obj['loss'], model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_obj['mpjpe']


    # Define train step and validation step functions
    @tf.function
    def train_step(input, gt):
        return step(input, gt, 'train')
        
    @tf.function
    def val_step(input, gt):
        return step(input, gt, 'val')

    if len(gpus) > 1:
        @tf.function
        def distributed_train_step(input, gt):
            per_replica_losses = strategy.run(train_step, args=(input, gt))

        @tf.function
        def distributed_val_step(input, gt):
            per_replica_losses = strategy.run(val_step, args=(input, gt))
    
    # Main Training Loop
    for epoch in range(start_epoch, cfg.EPOCH+1):
        metric_reset(metric_obj)
        train_start = time.time()
        for input, gt, info in train_ds:
            if len(gpus) > 1:
                distributed_train_step(input, gt)
            else:
                loss_obj = train_step(input, gt)
        
        train_end = time.time()

        for input, gt, info in val_ds:
            if len(gpus) > 1:
                distributed_val_step(input, gt)
            else:
                loss_obj = val_step(input, gt)

        val_end = time.time()
        
        reporter(metric_obj, epoch, train_time=(train_end - train_start), val_time=(val_end - train_end), ONLY_LOSS=ONLY_LOSS)
        logger.log_tensorboard(epoch, optimizer, metric_obj, cfg)
        logger.add_data(metric_obj)

        if epoch % cfg.SAVE_PERIOD == 0:
            logger.save_log(model, cfg.LOG_DIR, epoch)

if __name__ == '__main__':
    main()
    
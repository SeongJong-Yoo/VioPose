# ------------------------------------------------------------------------------
# This class is influenced by 'metric_history.py' 
# at https://github.com/goldbricklemon/uplift-upsample-3dhpe/tree/main 
#
# TODO
# ------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import os
import shutil

def remove_folder(dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            try:
                file_path = os.path.join(root, f)
                os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' %(file_path, e))
                break

        for d in dirs:
            try:
                dir_path = os.path.join(root, d)
                shutil.rmtree(dir_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' %(dir_path, e))
                break

    try:
        shutil.rmtree(dir)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' %(dir, e))

def check_path(prev_dir, dir, epoch, type, exe):
    if prev_dir is not None:
        os.remove(prev_dir)

    if not os.path.exists(dir):
        os.mkdir(dir)
    dir = os.path.join(dir, f"{type}_{epoch:04d}.{exe}")

    return dir
    

def save_model(model, prev_dir, model_dir, epoch, type):
    model_dir = check_path(prev_dir, model_dir, epoch, type, 'h5')
    model.save_weights(model_dir)
    print(type + ' Model is Saved at ' + model_dir)

    return model_dir

def save_metric(history, prev_dir, metric_dir, epoch, type):
    metric_dir = check_path(prev_dir, metric_dir, epoch, type, 'npy')
    np.save(metric_dir, history)
    print(type + 'Model Metric is Saved at ' + metric_dir)

    return metric_dir


class Logger:
    def __init__(self, metrics):
        self.tb = None
        self.history = dict()
        self.metrics = metrics
        self.last_model_dir = None
        self.last_metric_dir = None
        self.best_model_dir = None
        self.best_metric_dir = None
        self.best_score = 9999

        self._make_metrics()

    def __del__(self):
        self.tb.close()

    def make_tensorboard(self, dir):
        tb_dir = os.path.join(dir, "tensorboard")
        self.tb = tf.summary.create_file_writer(tb_dir)

    def log_tensorboard(self, epoch, optimizer, metric_obj, cfg):
        with self.tb.as_default():
            tf.summary.scalar('train/LR', optimizer.lr, step=epoch)
            for _, key in enumerate(metric_obj):
                label = key.split('_')
                tf.summary.scalar(label[0] + '/'+ label[1], metric_obj[key].result(), step=epoch)

    def add_data(self, metric_obj):
        for metric, value in metric_obj.items():
            self.history[metric].append(value.result().numpy())

    def save_log(self, model, dir, epoch):
        model_dir = os.path.join(dir, "model")
        self.last_model_dir = save_model(model, self.last_model_dir, model_dir, epoch, type='last_model')

        metric_dir = os.path.join(dir, "metric")
        self.last_metric_dir = save_metric(self.history, self.last_metric_dir, metric_dir, epoch, type='last_model')

        if self.best_score > self.history['val_loss'][-1]:
            self.best_score = self.history['val_loss'][-1]
            self.best_model_dir = save_model(model, self.best_model_dir, model_dir, epoch, type='best_model')
            self.best_metric_dir = save_metric(self.history, self.best_metric_dir, metric_dir, epoch, type='best_model')
        if epoch % 100==0:
            save_model(model, None, model_dir, epoch, type='epoch_model')


    def save_cfg(self, cfg):
        dir = cfg.LOG_DIR + '/cfg_parameters.yaml'
        with open(dir, 'w') as f:
            print(cfg, file=f)

    def update_logger(self, history, model_dir, metric_dir):
        self.last_metric_dir = metric_dir
        self.last_model_dir = model_dir
        self.history = history

    def _make_metrics(self):
        for i, type in enumerate(self.metrics):
            train_type = 'train_'+ type
            val_type = 'val_' + type
            self.history[train_type] = list()
            self.history[val_type] = list()


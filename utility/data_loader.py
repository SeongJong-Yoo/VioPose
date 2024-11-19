import numpy as np
import tensorflow as tf
from utility.utils import pick_upper_body
import math
import random
from copy import deepcopy
import scipy.signal as signal
# import librosa

def low_pass_kp_2d(data, N, fs, cutoff):
    ''' 
    Discrete Low pass filter
    N: Order
    fs: Data sampling frequency
    cutoff: cutoff frequency
    '''
    output = []
    b, a = signal.butter(N, cutoff, fs=fs, btype='low', analog=False)
    frame, joint, dim = data.shape
    for i in range(joint):
        output.append(np.array([
                                signal.filtfilt(b, a, data[:, i, 0]),
                                signal.filtfilt(b, a, data[:, i, 1])
                                ]))
    output = np.array(output).transpose(2, 0, 1)
    return output

def keypoints_remap(kp):
    """ 
    Input:
        0: head, 1: neck, 2, RShoulder, 3: RElbow, 4: RWrist, 5: LShoulder, 6:LElbow, 7: LWrist, 8: MidHip, 9: RHip, 10: RKnee, 11: RAnkle
        12: LHip, 13: LKnee, 14: LAnkle, 15: REye, 16: LEye, 17: REar, 18: LEar, 19: LBigToe, 20: LSmallToe, 21: LHeel, 22: RBigToe, 23: RSmallToe, 24: RHeel
        25: LHand, 26: RHand
    Output:
        Body Joints: 14 joints (frame, 14, 3)
            Left:  0: Toe, 3: Knee, 2: Pelvis, 4: Shoulder,  1: Elbow, 5: Wrist, 12: Hand
            Right: 6: Toe, 9: Knee, 8: Pelvis, 10: Shoulder, 7: Elbow, 11: Wrist, 13: Hand

    """
    if len(kp.shape)==4:
        output = np.stack([
            kp[:, :, 19],
            kp[:, :, 6],
            kp[:, :, 12],
            kp[:, :, 13],
            kp[:, :, 5],
            kp[:, :, 7],
            kp[:, :, 22],
            kp[:, :, 3],
            kp[:, :, 9],
            kp[:, :, 10],
            kp[:, :, 2],
            kp[:, :, 4],
            kp[:, :, 25],
            kp[:, :, 26]
        ], axis=2)
        return output
    elif len(kp.shape)==3:
        output = np.stack([
            kp[:, 19],
            kp[:, 6],
            kp[:, 12],
            kp[:, 13],
            kp[:, 5],
            kp[:, 7],
            kp[:, 22],
            kp[:, 3],
            kp[:, 9],
            kp[:, 10],
            kp[:, 2],
            kp[:, 4],
            kp[:, 25],
            kp[:, 26]
        ], axis=1)
        return output

def flatten_last_two_dim(a):
    return a.reshape(a.shape[0],a.shape[1],-1)

def heap_centered(input):
    shape = input.shape
    input = np.transpose(input, [1, 0, 2])   # (joint, frame, dim)

    center = (input[2] + input[8]) / 2

    output = input - center[np.newaxis, :, :]

    return np.transpose(output, [1, 0, 2])  # (frame, joint, dim)

class ViolinDataset:
    def __init__(self, cfg, train, data_type='violin'):
        if cfg.MODEL.NUM_KEYPOINTS==8:
            pick_upper_only = True
        else:
            pick_upper_only = False

        if data_type=='in-the-wild':
            self.person_key={'train': [],
                             'test': [20, 21, 22, 23, 24, 25]}
            print("Load in-the-wild dataset")
        elif data_type=='violin':
            self.person_key = {'train': [ 4, 6, 7, 9, 10, 11, 12, 14, 15 ],
                                'test': [ 5, 8, 13 ]                
                                # 'test': [ 4 ]                 
                            }
            print("Load violin dataset")
        elif data_type=='youtube':
            self.person_key = {'train' : [],
                               'test': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
            print("Load youtube dataset")
        else:
            print("##### Data Load ERROR at ViolinDataset Class: Non supporting input type: ", data_type)
            exit()
        self.val_ratio = cfg.DATA.SPLIT_RATIO
        self.audio_dim = None
        self.audio_frame = None
        self.filters = ['P05C01T28',
                        'P08C08T13',
                        'P08C08T15',
                        'P10C07T10',
                        'P12C10T09',
                        'P08C01T23',
                        'P08C02T23',
                        'P08C07T23',
                        'P08C08T23']

        data = np.load(cfg.DATA.DIR, allow_pickle=True)
        data_audio = np.load(cfg.DATA.DIR.split('.')[0] + '.' + cfg.DATA.DIR.split('.')[1] + '_audio.npz', allow_pickle=True)

        if cfg.DATA.POSE_TYPE=='gt':
            keypoints = data['kp_optim_2d_vid'][()]
            for key, item in keypoints.items():
                keypoints[key] = np.asarray(item)
        elif cfg.DATA.POSE_TYPE=='openpose':
            keypoints = data['openpose_2d'][()]
            for key, item in keypoints.items():
                kp = np.asarray(item)[:, :, :2]
                if kp.shape[1] > 17:
                    kp = keypoints_remap(kp)
                keypoints[key] = kp
        elif cfg.DATA.POSE_TYPE=='mediapipe':
            keypoints = data['mediapipe_2d'][()]
        else:
            print("##### Data Load ERROR at ViolinDataset Class: Non supporting input type: ", cfg.DATA.POSE_TYPE)
            exit()

        if 'kp_optim_3d_vid' in data.keys():
            keypoints_optim = data['kp_optim_3d_vid'][()]
            keypoints2d_optim = data['kp_optim_2d_vid'][()]
        else:
            keypoints_optim = {}
            keypoints2d_optim = {}
            for key, item in keypoints.items():
                keypoints_optim[key] = np.zeros((item.shape[0], 14, 3))
                keypoints2d_optim[key] = np.zeros((item.shape[0], 14, 2))
        video_fps = data['kp_video_fps'][()]

        audio_fps_ref = 30
        audio_data = None
        if cfg.DATA.AUDIO_TYPE=='raw':
            audio_data = data_audio['audio_raw'][()]
            audio_fps_ref = 8000
        elif cfg.DATA.AUDIO_TYPE=='features':
            audio_data = data_audio['audio_data'][()]
            audio_fps_ref = 30
        elif cfg.DATA.AUDIO_TYPE=='high_features':
            audio_data = data_audio['audio_data_100FPS'][()]
            audio_fps_ref = 100
        elif cfg.DATA.AUDIO_TYPE=='high_300_features':
            audio_data = data_audio['audio_data_300FPS'][()]
            audio_fps_ref = 300
        elif cfg.DATA.AUDIO_TYPE=='spec':
            audio_data = data_audio['audio_spectrogram'][()]
            audio_fps_ref = 30
        elif cfg.DATA.AUDIO_TYPE=='spec_high':
            audio_data = data_audio['audio_spectrogram_100FPS'][()]
            audio_fps_ref = 100
        elif cfg.DATA.AUDIO_TYPE=='VQ_audio':
            if not 'VQ_audio' in data_audio:
                sound_stream_data = np.load(cfg.DATA.DIR.split('.')[0] + '.' + cfg.DATA.DIR.split('.')[1] + '_audio_VQ.npz', allow_pickle=True)
                audio_data = sound_stream_data['VQ_audio'][()]
                audio_fps_ref = 80
        elif cfg.DATA.AUDIO_TYPE=='MAE':
            if not 'mae' in data_audio:
                mae_data = np.load(cfg.DATA.DIR.split('.')[0] + '.' + cfg.DATA.DIR.split('.')[1] + '_audio_MAE.npz', allow_pickle=True)
                audio_data = mae_data['mae'][()]
                audio_fps_ref = 50

        audio_fps = deepcopy(video_fps)
        if not audio_fps_ref == 30:
            for key in audio_fps.keys():
                audio_fps[key]=audio_fps_ref

        ref_keys = keypoints_optim.keys()

        data_list = []
        for key in ref_keys:
            if data_type=='youtube':
                data_list.append(key)
                continue
            if key in self.filters:
                continue
            if train:
                if int(key[1:3]) in self.person_key['train']:
                    data_list.append(key)
            else:
                if int(key[1:3]) in self.person_key['test']:
                    data_list.append(key)
        
        self._data={'keypoints': {},
                    'kp_optim_3d_vid': {},
                    'kp_optim_2d_vid': {},
                    'kp_video_fps': {}}
        
        self.audio_process = True
        if not audio_data is None:
            self._data['audio'] = {}
            self._data['audio_fps'] = {}
            self._data['audio_fps_ref'] = audio_fps_ref

            if audio_fps_ref==30:
                self.audio_frame = cfg.MODEL.OUTPUT_FRAME
            else:
                self.audio_frame = math.floor(cfg.MODEL.OUTPUT_FRAME * audio_fps_ref / cfg.MODEL.VIDEO_FPS)
        else:
            self.audio_process = False

        for key in ref_keys:
            if not key in data_list:
                continue
            if key in self.filters:
                continue
            if pick_upper_only:
                self._data['keypoints'][key] = pick_upper_body(keypoints[key], True)
                self._data['kp_optim_3d_vid'][key] = pick_upper_body(np.asarray(keypoints_optim[key]), True)
                self._data['kp_optim_2d_vid'][key] = pick_upper_body(np.asarray(keypoints2d_optim[key]), True)
            else:
                self._data['keypoints'][key] = heap_centered(np.asarray(keypoints[key]))
                self._data['kp_optim_3d_vid'][key] = heap_centered(np.asarray(keypoints_optim[key]))
                self._data['kp_optim_2d_vid'][key] = heap_centered(np.asarray(keypoints2d_optim[key]))

            if not audio_data is None:
                self.audio_dim = audio_data[key].shape[-1]
                if audio_data[key].shape[-1]==36:
                    self._data['audio'][key] = audio_data[key][:, :-1]
                else:
                    self._data['audio'][key] = audio_data[key]
                self._data['audio_fps'][key] = audio_fps[key]
            self._data['kp_video_fps'][key] = video_fps[key]

    def __getitem__(self, key):
        return self._data[key]

class ChunkedGeneratorViolin:
    def __init__(self, cfg, batch_size, dataset, chunk_length, pad=0,
                 random_seed=1234, test=False):
        self.pairs = []
        self.saved_index = {}
        # start_index = 0
        self.pad = pad
        self.audio_process = dataset.audio_process
        self.receptive_size = cfg.MODEL.OUTPUT_FRAME
        self.test = test

        compensate = int((self.receptive_size - chunk_length) / 2)
        for key, value in dataset['keypoints'].items():
            n_chunks = math.ceil(value.shape[0] / chunk_length)
            bounds = np.arange(n_chunks + 1) * chunk_length
            keys = np.tile(key, (len(bounds - 1), 1))
            self.pairs += list(zip(keys, bounds[:-1] - compensate, bounds[1:] + compensate))

        self.ref_fps = cfg.MODEL.VIDEO_FPS
        self.span = self.receptive_size / self.ref_fps
        self.audio_frame = dataset.audio_frame

        self.batch_3d = np.empty((batch_size, chunk_length, dataset['kp_optim_3d_vid'][key].shape[-2], dataset['kp_optim_3d_vid'][key].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length + 2 * self.pad, dataset['kp_optim_2d_vid'][key].shape[-2], dataset['kp_optim_2d_vid'][key].shape[-1]))
        self.batch_kp = np.empty((batch_size, chunk_length + 2 * self.pad, dataset['keypoints'][key].shape[-2], dataset['keypoints'][key].shape[-1]))

        if self.audio_process:
            if cfg.DATA.AUDIO_TYPE=='raw':
                self.batch_audio = np.empty((batch_size, self.audio_frame))
            else:
                self.batch_audio = np.empty((batch_size, self.audio_frame, dataset['audio'][key].shape[-1]))

        self.batch_size = batch_size
        self.random_seed = random_seed
        self.dataset = dataset

        if self.test:
            self.val_pairs = self.pairs
        else:
            self.pairs = self.pairs

            
    def divide_train_val(self, val_ratio):
        num_val = int(len(self.pairs) * val_ratio)
        random.seed(self.random_seed)
        random.shuffle(self.pairs)
        self.val_pairs = self.pairs[:num_val]
        self.train_pairs = self.pairs[num_val:]

    def next_pairs(self, train=True):
        if train:
            random.seed(self.random_seed)
            random.shuffle(self.train_pairs)
            pairs = self.train_pairs
        else:
            pairs = self.val_pairs
        return pairs
    
    def get_batch(self, seq_i, start_3d, end_3d):
        self.batch_2d = None
        self.batch_3d = None
        self.batch_kp = None
        self.batch_audio = None
        self.batch_up = None

        subject = seq_i
        seq_name = subject[0]

        fps = self.dataset['kp_video_fps'][seq_name]
        start_time = start_3d / fps

        seq_kp = self.dataset['keypoints'][seq_name]
        seq_2d = self.dataset['kp_optim_2d_vid'][seq_name]
        seq_3d = self.dataset['kp_optim_3d_vid'][seq_name]

        high_3d = min(end_3d, seq_kp.shape[0])
        pad_right_3d = end_3d - high_3d
        low_3d = max(0, start_3d)
        pad_left_3d = low_3d - start_3d
        
        self.batch_2d = np.pad(seq_2d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
        self.batch_3d = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
        self.batch_kp = np.pad(seq_kp[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')

        if self.audio_process:
            audio_fps = self.dataset['audio_fps'][seq_name]
            seq_audio = self.dataset['audio'][seq_name]
            len_audio = seq_audio.shape[0]

            audio_start = int(start_time * audio_fps)
            if audio_start > len_audio - 1:
                audio_start = len_audio - 1

            audio_end = audio_start + self.audio_frame
            high_audio = min(audio_end, seq_audio.shape[0])
            pad_right_audio = audio_end - high_audio
            low_audio = max(0, audio_start)
            pad_left_audio = low_audio - audio_start

            self.batch_audio = np.pad(seq_audio[low_audio:high_audio], ((pad_left_audio, pad_right_audio), (0, 0)), 'edge')

        return self.batch_kp, self.batch_3d, self.batch_2d, self.batch_audio

class Fusion:
    def __init__(self, cfg, dataset:ViolinDataset, generator:ChunkedGeneratorViolin, train=True):
        self.stride = cfg.DATA.STRIDE
        self.dataset = dataset
        self.pad = cfg.DATA.PAD
        self.train = train

        self.generator = generator
        self.pairs = self.generator.next_pairs(self.train)
        if train:
            print('INFO: Training on {} frames'.format(self.__len__()))
        else:
            print('INFO: Validating on {} frames'.format(self.__len__()))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        seq_name, start, end = self.pairs[idx]

        kp, kp_3d, kp_2d, audio = self.generator.get_batch(seq_name, start, end)
        
        input = {"pose": kp}
        if self.dataset.audio_process:
            input['audio'] = audio

        output={"pose": kp_3d,
                "pose_2d": kp_2d}
        
        info = {'trial': seq_name[0],
                'start': start,
                'end': end,
                'fps': self.dataset['kp_video_fps'][seq_name[0]]
                }

        return input, output, info
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.pairs = self.generator.next_pairs(self.train)

def slice_generate_dataset(dataset:ViolinDataset, cfg, test=False):
    gen = ChunkedGeneratorViolin(cfg, cfg.BATCH_SIZE,
                                 dataset, cfg.DATA.STRIDE, pad=cfg.DATA.PAD, test=test)
    
    if not test:
        gen.divide_train_val(cfg.DATA.SPLIT_RATIO)

        data_train = Fusion(cfg, dataset, gen, train=True)
        data_val = Fusion(cfg, dataset, gen, train=False)
    else:
        data_val= Fusion(cfg, dataset, gen, train=False)

    seq_name, start, end = data_val.pairs[0]
    kp, kp_3d, kp_2d, audio = data_val.generator.get_batch(seq_name, start, end)

    output_signature = (
            {'pose': tf.TensorSpec(shape=kp.shape, dtype=tf.float32)},
            {'pose': tf.TensorSpec(shape=kp_3d.shape, dtype=tf.float32),
             'pose_2d': tf.TensorSpec(shape=kp_2d.shape, dtype=tf.float32)},
            {'trial': tf.TensorSpec(shape=(), dtype=tf.string),
             'start': tf.TensorSpec(shape=(), dtype=tf.int32),
             'end': tf.TensorSpec(shape=(), dtype=tf.int32),
             'fps': tf.TensorSpec(shape=(), dtype=tf.float32)}
    )
    
    if not audio is None:
        output_signature[0]['audio'] = tf.TensorSpec(shape=audio.shape, dtype=tf.float32)

    if not test:
        train_ds = tf.data.Dataset.from_generator(data_train, output_signature=output_signature)

        train_ds = train_ds.batch(cfg.BATCH_SIZE)
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        train_ds = None
        
    val_ds = tf.data.Dataset.from_generator(data_val, output_signature=output_signature)
    val_ds = val_ds.batch(cfg.BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    audio_dim = dataset.audio_dim
    audio_frame = dataset.audio_frame

    return train_ds, val_ds, audio_frame, audio_dim
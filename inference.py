import argparse
import numpy as np
import os
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from matplotlib import pyplot as plt
import natsort

from utility.loss import *
from utility.utils import *
from utility.logger import *
from config import cfg
from config import read_config
from main import build_model

connected_mediapipe = [
    (0, 3), (3, 2), (2, 4), (2, 8), (6, 9), (9, 8), (8, 10),
    (4, 10), (1, 4), (1, 5), (5, 13), (5, 14), (13, 12), (5, 12),
    (10, 7), (7, 11), (11, 17), (11, 16), (16, 15), (11, 15)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description='MAPNet Argument Parser')
    parser.add_argument('--folder',
                        help='Experiment folder name',
                        required=True,
                        type=str)
    parser.add_argument('--video_path',
                        default='best',
                        help='/VIDEO/PATH.mp4',
                        type=str)
    parser.add_argument('--audio_path',
                        default=None,
                        help='/AUDIO/PATH.wav',
                        type=str)
    parser.add_argument('--vis',
                        type=lambda x: x.lower() == 'true',
                        default=True,
                        help='Visualize the result (--vis False to disable)')
    args = parser.parse_args()

    return args


def mediapipe_kp_remap(kp):
    """
    Input:
        Body Joints: 33 joints (frame, 33, 3)
        0 - nose, 1 - left eye (inner), 2 - left eye, 3 - left eye (outer), 4 - right eye (inner), 5 - right eye,
        6 - right eye (outer), 7 - left ear, 8 - right ear, 9 - mouth (left), 10 - mouth (right), 11 - left shoulder,
        12 - right shoulder, 13 - left elbow, 14 - right elbow, 15 - left wrist, 16 - right wrist, 17 - left pinky,
        18 - right pinky, 19 - left index, 20 - right index, 21 - left thumb, 22 - right thumb, 23 - left hip,
        24 - right hip, 25 - left knee, 26 - right knee, 27 - left ankle, 28 - right ankle, 29 - left heel,
        30 - right heel, 31 - left foot index, 32 - right foot index

    Output:
        Body Joints: 18 joints (frame, 18, 3)
        Left:  0: Ankle, 3: Knee, 2: Pelvis, 4: Shoulder,  1: Elbow, 5: Wrist, 12: Index, 13: Pinky, 14: Thumb
        Right: 6: Ankle, 9: Knee, 8: Pelvis, 10: Shoulder, 7: Elbow, 11: Wrist, 15: Index, 16: Pinky, 17: Thumb
    """
    output = np.stack([
        kp[:, 27],  # 0 - left ankle
        kp[:, 13],  # 1 - left elbow
        kp[:, 23],  # 2 - left pelvis
        kp[:, 25],  # 3 - left knee
        kp[:, 11],  # 4 - left shoulder
        kp[:, 15],  # 5 - left wrist
        kp[:, 28],  # 6 - right ankle
        kp[:, 14],  # 7 - right elbow
        kp[:, 24],  # 8 - right pelvis
        kp[:, 26],  # 9 - right knee
        kp[:, 12],  # 10 - right shoulder
        kp[:, 16],  # 11 - right wrist
        kp[:, 19],  # 12 - left index
        kp[:, 17],  # 13 - left pinky
        kp[:, 21],  # 14 - left thumb
        kp[:, 20],  # 15 - right index
        kp[:, 18],  # 16 - right pinky
        kp[:, 22],  # 17 - right thumb
    ], axis=1)
    return output

def fill_empty(kp):
    if len(kp) == 0:
        return 
    
    for i in range(len(kp)):
        if len(kp[i])==0:
            j=i
            while len(kp[j])==0:
                j+=1
            if j < len(kp):
                for k in range(j-i):
                    kp[i+k]=kp[j]
            else:
                j=i
                while len(kp[j])==0:
                    j-=1
                for k in range(i-j):
                    kp[i-k]=kp[i]
    return kp


def mediapipe_extraction(video_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    data = {}
    data['mediapipe_2d'] = []
    data['kp_video_fps'] = []
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_videos = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keypoints2d = np.empty(shape=(num_videos, 33, 2))
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        number_frames = 0
        while True:
            success, image = cap.read()
            if not success:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results_2d = detector.detect(mp_image)
            if not results_2d.pose_landmarks:
                number_frames += 1
                continue

            keypoint_2d = []
            shape = image.shape
            for j in range(len(results_2d.pose_landmarks)):
                for i in range(len(results_2d.pose_landmarks[j])):
                    keypoint_2d.append([results_2d.pose_landmarks[j][i].x * shape[1], results_2d.pose_landmarks[j][i].y * shape[0]])
            keypoints2d[number_frames]=np.asarray(keypoint_2d)
            number_frames += 1

    keypoints2d = fill_empty(keypoints2d)
    data['mediapipe_2d'] = mediapipe_kp_remap(keypoints2d)
    data['kp_video_fps'] = fps
    return data

def process_audio_file(audio_name, FPS = 50, HOP_LENGTH = 512):
    SR = FPS * HOP_LENGTH
    waveform, detected_sr = librosa.load(audio_name, mono=True, sr=SR)
  
    # References
    # https://stackoverflow.com/questions/63240852/shape-of-librosa-feature-melspectrogram
    # ** https://stackoverflow.com/questions/62727244/what-is-the-second-number-in-the-mfccs-array/62733609#62733609
    # L = Length of the signal in frames (not samples)
    # N = n_mels
    # Output shape = (L, N)
    # Eg. 2005.wav, L = 1 + len(waveform)//hop_length ie. 815
    # Can also get the same by round( (SR / hop_length) * time_in_sec )
    envelope = librosa.onset.onset_strength(y=waveform, sr=SR)
    peak_idxs = librosa.onset.onset_detect(onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0

    chroma_cens = librosa.feature.chroma_cens(y=waveform, sr=SR, n_chroma=12).T

    mfcc = librosa.feature.mfcc(y=waveform, sr=SR, n_mfcc=20).T
    rms = librosa.feature.rms(y=waveform)

    f1 = envelope[:, None]
    f2 = mfcc
    f3 = chroma_cens
    f4 = peak_onehot[:, None]
    f5 = rms.T
    # NORMALIZE Features that are not between -1.0 and 1.0 already
    f1_norm = librosa.util.normalize(f1)
    f2 = librosa.util.normalize(f2)
    f5 = librosa.util.normalize(f5)

    len = min(f1.shape[0], f2.shape[0], f3.shape[0], f4.shape[0], f5.shape[0])


    audio_feature = np.concatenate([f1_norm[:len,:], f2[:len,:], f3[:len,:], f4[:len,:] , f5[:len,:], f1[:len,:]], axis=-1)

    return audio_feature

def match_audio_feature(audio_path, FPS, HP, ref_len):
    init_FPS = FPS
    if FPS > 50:
        gap = 0.01
    else:
        gap = 0.005
    visit = [False, False]
    audio_feature = process_audio_file(audio_name=audio_path, FPS=FPS, HOP_LENGTH=HP)
    if ref_len < len(audio_feature):
        FPS = FPS - gap
        while True:
            audio_feature = process_audio_file(audio_name=audio_path, FPS=FPS, HOP_LENGTH=HP)

            if len(audio_feature) == ref_len:
                return audio_feature, FPS
            elif ref_len > len(audio_feature):
                visit[0] = True
                if visit[0] and visit[1]:
                    gap = gap / 2
                FPS = FPS + gap
            else:
                visit[1] = True
                if visit[0] and visit[1]:
                    gap = gap / 2
                FPS = FPS - gap
            if init_FPS - FPS > 3:
                print("File: ", audio_path, ", Adjusted FPS: ", FPS)
                exit()

    elif ref_len > len(audio_feature):
        FPS = FPS + gap
        while True:
            audio_feature = process_audio_file(audio_name=audio_path, FPS=FPS, HOP_LENGTH=HP)

            if len(audio_feature) == ref_len:
                return audio_feature, FPS
            elif ref_len > len(audio_feature):
                visit[0] = True
                if visit[0] and visit[1]:
                    gap = gap / 2
                FPS = FPS + gap
            else:
                visit[1] = True
                if visit[0] and visit[1]:
                    gap = gap / 2
                FPS = FPS - gap
            if init_FPS - FPS > 3:
                print("File: ", audio_path, ", Adjusted FPS: ", FPS)
                exit()
    else:
        if init_FPS - FPS > 3:
            print("File: ", audio_path, ", Adjusted FPS: ", FPS)
        return audio_feature, FPS
    
def audio_feature_extraction(audio_path, vid_FPS, kp):
    HOP_LENGTH = 512
    audio_FPS = 100
    ref_len = int(len(kp) * audio_FPS / vid_FPS)
            
    audio_feature, audio_FPS = match_audio_feature(audio_path, audio_FPS, HOP_LENGTH, ref_len)

    print(f"Audio features are extracted with audio_FPS: {audio_FPS}")
    return audio_feature


def process_video(video_path, audio_path):
    print("######### Extracting 2D keypoints using Mediapipe #########")
    data = mediapipe_extraction(video_path)
    if audio_path != None:
        kp = data['mediapipe_2d']
        vid_FPS = data['kp_video_fps']
        print("######### Extracting audio features #########")
        data['audio_data_100FPS'] = audio_feature_extraction(audio_path, vid_FPS, kp)
    
    return data

def slice_dataset(dataset, cfg, MUSIC):
    audio_frame = None
    audio_dim = None
    fps = dataset['kp_video_fps']
    num_chunks = math.ceil(len(dataset['mediapipe_2d']) / cfg.DATA.STRIDE)
    output = {"mediapipe_2d": [dataset['mediapipe_2d'][:cfg.DATA.STRIDE][np.newaxis, ...]],
              "kp_video_fps": [dataset['kp_video_fps']],
              "info":[np.concatenate((np.zeros(cfg.DATA.STRIDE), [1]))]}
    
    if MUSIC:
        audio_frame = cfg.MODEL.AUDIO_MODULE.FRAME
        audio_dim = dataset['audio_data_100FPS'].shape[1]
        output["audio_data_100FPS"] = [dataset['audio_data_100FPS'][:audio_frame][np.newaxis, ...]]

    for i in range(1, num_chunks):
        start_time = i * cfg.DATA.STRIDE / fps
        start_frame = i * cfg.DATA.STRIDE
        end_frame = (i + 1) * cfg.DATA.STRIDE
        high = min(end_frame, len(dataset['mediapipe_2d']))
        pad_right = end_frame - high
        next_kp_2d = np.pad(dataset['mediapipe_2d'][start_frame:high], ((0, pad_right), (0, 0), (0, 0)), 'edge')
        output["mediapipe_2d"].append(next_kp_2d[np.newaxis, ...]) #  = np.concatenate((output["mediapipe_2d"], ), axis=0)

        if pad_right > 0:
            padded = np.ones(pad_right + 1)
            next_info = np.concatenate((np.zeros(cfg.DATA.STRIDE - pad_right), padded), axis=0)
        else:
            next_info = np.concatenate((np.zeros(cfg.DATA.STRIDE), [1]))
        output["info"].append(next_info) # = np.concatenate((output["info"], info), axis=0)

        if MUSIC:
            audio_fps = 100
            len_audio = len(dataset['audio_data_100FPS'])
            audio_start = int(start_time * audio_fps)
            if audio_start > len_audio - 1:
                audio_start = len_audio - 1

            audio_end = audio_start + audio_frame
            high_audio = min(audio_end, len_audio)
            pad_right_audio = audio_end - high_audio

            next_audio_data = np.pad(dataset['audio_data_100FPS'][audio_start:high_audio], ((0, pad_right_audio), (0, 0)), 'edge')
            output["audio_data_100FPS"].append(next_audio_data[np.newaxis, ...]) # = np.concatenate((output["audio_data_100FPS"], next_audio_data), axis=0)

    return output, audio_frame, audio_dim

def merge(envelope, KALMAN_FILTER):
    output = {'pred': [],
              'input': []}
    
    if KALMAN_FILTER:
        output['acc'] = []
        output['vel'] = []

    for i in range(len(envelope['pred'])):
        index_pad = np.where(envelope['info_list'][i] == 1)[0][0]
        pred = envelope['pred'][i][:index_pad]
        input = envelope['input'][i][:index_pad]

        if KALMAN_FILTER:
            acc = envelope['acc'][i][:index_pad]
            vel = envelope['vel'][i][:index_pad]
        if i==0:
            output['pred'] = pred
            output['input'] = input
            if KALMAN_FILTER:
                output['acc'] = acc
                output['vel'] = vel
        else:
            output['pred'] = np.concatenate((output['pred'], pred), axis=0)
            output['input'] = np.concatenate((output['input'], input), axis=0)
            if KALMAN_FILTER:
                output['acc'] = np.concatenate((output['acc'], acc), axis=0)
                output['vel'] = np.concatenate((output['vel'], vel), axis=0)

    return output

def heap_centered(input):
    input = np.squeeze(input)
    input = np.transpose(input, [1, 0, 2])   # (joint, frame, dim)

    center = (input[2] + input[8]) / 2

    output = input - center[np.newaxis, :, :]

    return np.transpose(output, [1, 0, 2])[np.newaxis, ...]  # (frame, joint, dim)


def evaluate(model, test_ds, MUSIC, KALMAN_FILTER):
    envelope = {'pred': [],
                'input': [],
                'info_list': []}

    if KALMAN_FILTER:
        envelope['acc'] = []
        envelope['vel'] = []

    num_chunk = len(test_ds['mediapipe_2d'])
    for i in range(num_chunk):
        pose_x = heap_centered(test_ds['mediapipe_2d'][i])
        if MUSIC:
            audio_x = test_ds['audio_data_100FPS'][i]
        else:
            audio_x = np.zeros_like(pose_x)

        output = model([pose_x, audio_x], training=False)

        envelope['pred'].append(np.squeeze(output['spatial'][1].numpy()))
        envelope['input'].append(np.squeeze(test_ds['mediapipe_2d'][i]))
        envelope['info_list'].append(test_ds['info'][i])

        if KALMAN_FILTER:
            envelope['acc'].append(np.squeeze(output['acc'].numpy()))
            envelope['vel'].append(np.squeeze(output['vel'].numpy()))

    return merge(envelope, KALMAN_FILTER)

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

def make_video(data, video_path, root_path, fps):
    save_path = os.path.join(root_path, 'output.mp4')
    
    kp = data['input']
    kp_3d = data['pred']
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dpi = 300
    fig  = plt.figure(dpi=dpi, figsize=(2*width/dpi, height/dpi))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.axis("off")

    fig.canvas.draw()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (2*width, height))

    counter = 0
    for i in tqdm(range(video_length)):
        plt.clf()

        ret, img = cap.read()

        ax = plt.subplot(1, 2, 1)       
        ax.imshow(img)            

        ax.scatter(kp[i][:, 0], kp[i][:, 1], color='red', s=2, alpha=1)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        if kp.shape[1]==18:
            connection = connected_mediapipe
        else:
            connection = None
        if not connection is None: 
            for joint1, joint2 in connection:
                ax.plot([kp[i][joint1, 0], kp[i][joint2, 0]], 
                        [kp[i][joint1, 1], kp[i][joint2, 1]], 'red', linewidth=1)

        ax = plt.subplot(1, 2, 2, projection='3d')  # Create a subplot for 3D keypoints
        ax.scatter(kp_3d[i][:, 0], kp_3d[i][:, 1], kp_3d[i][:, 2], color='blue', s=2, alpha=1)
        if connection is not None:
            for joint1, joint2 in connection:
                ax.plot([kp_3d[i][joint1, 0], kp_3d[i][joint2, 0]], 
                        [kp_3d[i][joint1, 1], kp_3d[i][joint2, 1]], 
                        [kp_3d[i][joint1, 2], kp_3d[i][joint2, 2]], 'blue', linewidth=1)
        # ax.add_artist
        # Add Axis information
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.grid(True)
        # ax.legend()
        limits = [1300, 1300, 650]
        x_min = np.min(kp_3d[i, :, 0])
        x_max = np.max(kp_3d[i, :, 0])
        y_min = np.min(kp_3d[i, :, 1])
        y_max = np.max(kp_3d[i, :, 1])
        z_min = np.min(kp_3d[i, :, 2])
        z_max = np.max(kp_3d[i, :, 2])
        x_diff = limits[0] - (x_max - x_min)
        y_diff = limits[1] - (y_max - y_min)
        z_diff = limits[2] - (z_max - z_min)

        ax.set_xlim3d(left=x_min - x_diff/2, right=x_max + x_diff/2)
        ax.set_ylim3d(bottom=y_min - y_diff/2, top=y_max + y_diff/2)
        ax.set_zlim3d(bottom=z_min - z_diff/2, top=z_max + z_diff/2)

        ax.view_init(280, -90)

        plt.tight_layout(pad=0)
        fig.canvas.draw()

        img = np.array(fig.canvas.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        video_writer.write(img)

    video_writer.release()
    plt.close(fig)


def combine_audio(video_path, audio_path, save_path):
    from moviepy.editor import VideoFileClip, AudioFileClip
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(os.path.join(save_path, "output_w_audio.mp4"), codec='libx264')


def main():
    KALMAN_FILTER=True
    MUSIC=True
    args = parse_args()
    if args.audio_path==None:
        args.folder = args.folder + '_wo_audio'
        cfg_path = os.path.join(args.folder, 'cfg_parameters.yaml')    
        MUSIC=False
        print("######### Using model without audio #########")
    else:
        cfg_path = os.path.join(args.folder, 'cfg_parameters.yaml')
        print("######### Using model with audio #########")
    read_config(cfg, cfg_path)
    cfg.defrost()
    cfg.MODE='test'
    cfg.DATA.FILTERED=False
    cfg.DATA.STRIDE=cfg.MODEL.OUTPUT_FRAME
    cfg.freeze()

    dataset = process_video(args.video_path, args.audio_path)

    test_ds, audio_frame, audio_dim = slice_dataset(dataset, cfg, MUSIC)

    # Load Model
    model, ONLY_LOSS = build_model(cfg, audio_frame, audio_dim)
    model = read_model(args.folder, "best", model)

    output = evaluate(model,test_ds, MUSIC, KALMAN_FILTER)

    if KALMAN_FILTER:
        output['pred']=kalman_filter(output['pred'], output['vel'], output['acc'])


    save_path = './output'
    if args.vis:
        print("######### Video saving at: ", save_path, " #########")
        # Generate Video
        make_video(output, args.video_path, save_path, dataset['kp_video_fps'])
        if args.audio_path!=None:
            combine_audio(os.path.join(save_path, 'output.mp4'), args.audio_path, save_path)

    result = {"input": output['input'],
              "pred": output['pred'],
              "fps": dataset['kp_video_fps']}
    
    np.save(os.path.join(save_path, 'result.npy'), result)


if __name__ == '__main__':
    main()
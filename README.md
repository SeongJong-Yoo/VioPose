
# VioPose 
This repository provides code and pretrained weights for the paper. The code is provided without any warranty. If using this code, please cite our work as shown below. For more information please visit our [project website](https://sj-yoo.info/viopose/) 

	@inproceedings{Hong_2021_ICCV,
    	author    = {Yoo, Seong Jong and Shrestha, Snehesh and Muresanu, Irina and Fermuller, Cornelia},
    	title     = {{VioPose}: Violin Performance 4D Pose Estimation by Hierarchical Audiovisual Inference},
    	booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    	year      = {2025},
	}

## Installation 
### Installation using Conda
```
conda create -n viopose python==3.9
conda activate viopose
conda install pip
pip install -r requirements.txt
./download.sh
```

- Install [tensorflow](https://www.tensorflow.org/install/pip) $\approx$ 2.13  and cuda accordingly
- Download dataset (will release soon) and locate it at `data/FullData` 

## Run Demo
### VioDat Test Demo (currently unavailable)
1. Run `test.py` script. This code will generate and save *result.npz* at `./Logs/VioPose`
```
python3 test.py --folder Logs/VioPose --data violin --data_path data/FullData/mmViolin_v1.0.npz
```
2.  Run `generate_video.py` script

### In-the-wild Demo
1. Prepare video (mp4 format) and audio (wav format). For demo we prepared video at `./data/demo` 
2. Run `inference.py` script
```
python3 inference.py --folder Logs/VioPose --video_path data/demo/demo.mp4 --audio_path data/demo/demo.wav --vis True
```
- If you don't provide audio file then *VioPose_wo_audio* model will be used for inference
- The output video is saved at `./output/output_w_audio.mp4` with audio and `./output/output.mp4` without audio

## Train from Scratch
1. Prepare *VioDat* at `./data/FullData` (currently unavailable)
2. Run `main.py` script
```
python3 main.py --cfg ./config/viopose.yaml > ./out_log/viopose.out
python3 test.py --folder /Logs/viopose --data violin --data_path data/FullData/mmViolin_v1.0.npz
```

## Bug Report
Please raise an issue on Github for issues related to this code. If you have any questions related about the code feel free to send an email to here (yoosj@umd.edu). 

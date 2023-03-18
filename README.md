# Implementation of [AAAI'23 Oral] "GLT-T: Global-Local Transformer Voting for 3D Single Object Tracking in Point Clouds"

## Introduction

## Setup
### Installation
+ Create the environment
  ```
  git clone https://github.com/haooozi/GLT-T.git
  cd GLT-T
  conda create -n GLT-T  python=3.7
  conda activate GLT-T
  ```
+ Install other dependencies:
  ```
  pip install -r requirement.txt
  ```
### KITTI dataset
+ Download the data for [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).
+ Unzip the downloaded files.
+ Put the unzipped files under the same folder as following.
  ```
  [Parent Folder]
  --> [calib]
      --> {0000-0020}.txt
  --> [label_02]
      --> {0000-0020}.txt
  --> [velodyne]
      --> [0000-0020] folders with velodynes .bin files
  ```

### NuScenes dataset
+ Download the dataset from the [download page](https://www.nuscenes.org/download)
+ Extract the downloaded files and make sure you have the following structure:
  ```
  [Parent Folder]
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    maps	        -	Folder for all map files: rasterized .png images and vectorized .json files.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
  ```
>Note: We following [Open3DSOT](https://github.com/Ghostish/Open3DSOT) for dataset setting. More details can be referred to it.

## Quick Start
### Training
To train a model (e.g., for the Car category), you must specify the `.yaml` file with `--cfg` argument.
```
python3.7 main.py  --cfg ./cfgs/Car_kitti.yaml
```
You can also use `CUDA_VISIBLE_DEVICES` to select specific GPUs.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.7 main.py  --cfg ./cfgs/Car_kitti.yaml
```
### Testing
To test a trained model (e.g., for the Car category), you must load spedific model with `--checkpoint` and add `--test` flag.
```
python3.7 main.py  --cfg ./cfgs/Car_kitti.yaml --checkpoint ./trained_model/Car_kitti.ckpt --test
```

### Visualization
Visualization code is integrated into `./models/base_model.py`, you can add `--track` to specific point cloud sequence.
```
python3.7 main.py  --cfg ./cfgs/Car_kitti.yaml --checkpoint ./trained_model/Car_kitti.ckpt --test --track 0/1/2...
```

## Citation
If you find GLT-T useful, please consider citing: 

```bibtex
@article{nie2022glt,
  title={GLT-T: Global-Local Transformer Voting for 3D Single Object Tracking in Point Clouds},
  author={Nie, Jiahao and He, Zhiwei and Yang, Yuxiang and Gao, Mingyu and Zhang, Jing},
  journal={arXiv preprint arXiv:2211.10927},
  year={2022}
}
```

## Acknowledgement
This repo builds on the top of [Open3DSOT](https://github.com/Ghostish/Open3DSOT).

Thank shanjiayao for his implementation of [PTT](https://github.com/shanjiayao/PTT), and Jasonkks for his implementation of [PTTR](https://github.com/Jasonkks/PTTR).

# TRAIN and EVAL


## Installation and Data preperation
Our code is based on [UniAD](https://github.com/OpenDriveLab/UniAD), we only provide the configuration files related to FusionAD. 

Please clone the source code of UniAD into this folder first.
```shell
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD && git checkout v1.0.1
```
Then follow the instructions for UniAD to [install the environment](https://github.com/OpenDriveLab/UniAD/blob/main/docs/INSTALL.md) and [prepare the dataset](https://github.com/OpenDriveLab/UniAD/blob/main/docs/DATA_PREP.md). 

Once everything is set up, the folder structure should look like this:

**The Overall Structure**

*Please make sure the structure of UniAD is as follows:*
```
FusionAD
├── projects/
├── tools/
├── fusionad_ckpts/
│   ├── epoch_29.pth
├── UniAD/
│   ├── data/
│   │   ├── nuscenes/
│   │   │   ├── can_bus/
│   │   │   ├── maps/
│   │   │   ├── samples/
│   │   │   ├── sweeps/
│   │   │   ├── v1.0-test/
│   │   │   ├── v1.0-trainval/
│   │   ├── infos/
│   │   │   ├── nuscenes_infos_temporal_train.pkl
│   │   │   ├── nuscenes_infos_temporal_val.pkl
│   │   ├── others/
│   │   │   ├── motion_anchor_infos_mode6.pkl
```

## Get the checkpoint
Checkpoint file can be found in [here](https://drive.google.com/file/d/1QYPw6L00DKTGJUljOJGzBGAzI0QUZRNa/view?usp=sharing)
```shell
mkdir ckpts && cd ckpts
wget https://drive.google.com/file/d/1QYPw6L00DKTGJUljOJGzBGAzI0QUZRNa/view?usp=sharing
```

## Evaluation Command <a name="example"></a>
### Evaluation Command
Please make sure you have prepared the environment and the nuScenes dataset.
```shell
cd FusionAD
# single GPU
./tools/dist_eval.sh ./projects/configs/stage2_e2e/fusion_base_e2e.py ./ckpts/fusion_latest.pth 1
# multi GPU
./tools/dist_eval.sh ./projects/configs/stage2_e2e/fusion_base_e2e.py ./ckpts/fusion_latest.pth 8
```

If everything is prepared properly, the output results should be:

```
NDS: 0.646
mAP: 0.574
amota: 0.502
amotp: 1.059
Motion Metric:
min_ade_err: 0.389
min_fde_err: 0.615
miss_rate_err: 0.084
IoU Metrics:
drivable_iou_mean: 0.732
lanes_iou_mean: 0.368
Occupancy Metrics:
VPQ-n: 64.951  VPQ-f+: 50.330
IoU-n: 70.540  IoU-f+: 50.985
Planning Metrics:
CR_avg: 0.116
L2: 0.708
CR_traj: 0.349
```

##  Train <a name="train"></a>
### Train Command
```shell
# N_GPUS is the number of GPUs used. Recommended >=8.
./tools/dist_train.sh ./projects/configs/stage2_e2e/fusion_base_e2e.py N_GPUS
```

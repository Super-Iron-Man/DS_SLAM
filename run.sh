#!/bin/bash

./build/bin/ds_slam_dataset \
    files=/home/user/zengjie/vslam_data/tum_monoVO/all_sequences/sequence_01/images.zip \
    calib=/home/user/zengjie/vslam_data/tum_monoVO/all_sequences/sequence_01/camera.txt \
    gamma=/home/user/zengjie/vslam_data/tum_monoVO/all_sequences/sequence_01/pcalib.txt \
    vignette=/home/user/zengjie/vslam_data/tum_monoVO/all_sequences/sequence_01/vignette.png \
    preset=0 \
    mode=0 \
    speed=0 \
    nogui=0 \
    nolog=0



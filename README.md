# DS_SLAM: Direct Sparse SLAM

DS_SLAM is based on DSO (https://github.com/JakobEngel/dso). I add the loop closing and make it as complete vSLAM system. Please have a look DSO's README (https://github.com/JakobEngel/dso/blob/master/README.md), and compile and run the system.

## Compile
System Dependencies:
1. DSO dependencies
2. Ceres Solver (http://ceres-solver.org/installation.html)

Install dependencies and compile the system: 
```
    cd DS_SLAM
    mkdir build
    cd build
    cmake ..
    make -j3
```
## Run
I run example on three dataset:  
 - TUM-Mono: https://vision.in.tum.de/mono-dataset
 - Kitti: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
 - EuRoC: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

run ds_slam_dataset, config DBoW file like "dbow=xxx" (file is under *Examples/DBoW*), other configuration is same as DSO.

## Notes
Improved points include:  
1. Add loop closing.
2. Povide loop detection with two feature, BRIEF and ORB.
3. Support two modes of loop closing and vo, separation mode and coupling mode.
4. Improve display fo pose graph.

## Related reference
1. VINS-Mono (https://github.com/HKUST-Aerial-Robotics/VINS-Mono)
2. LDSO (https://github.com/tum-vision/LDSO)

## License
DS_SLAM, like reference SLAM system, is licensed under GPLv3. Dependent third-party libraries include MIT, DSB, GPL etc.
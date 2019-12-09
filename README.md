*This is a fork of [VisualComputingInstitute/ShapePriors_GCPR16](https://github.com/VisualComputingInstitute/ShapePriors_GCPR16) which adds evaluation on the KITTI Object benchmark*. The old Readme is reproduced below.

### Setup:
+ Clone: `git clone --recursive https://github.com/lkskstlr/ShapePriors_GCPR16.git`
+ Install dependencies in `external` use `opencv-2.4.13.6`, `VTK-7.1.1`
+ Install other dependencies listed in the original README
+ Build the `external/viz` library as in the original README
+ Build this library `(mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DVIZ_LIBRARY=../external/viz/build/libVIZ.so && make -j)`
+ Build the evaluation tool `(cd kitti_eval && g++ -std=c++11 -O2 -o evaluate_object evaluate_object_3d_offline.cpp)`


### Run:
Assume that `base` holds the path to the standard `kitti/object/training` folder, then
+ Make results folder: `mkdir -p data/kitti_object/result/data`
+ Generate disparity maps in `$base/disp_2`
+ Generate ground-planes in `$base/plane`
+ To run with the groundtruth labels as initial detections: `cd build && ./kitti_object $base/image_2 $base/label_2 $base/disp_2 $base/calib $base/plane ../data/kitti_object/result/data`
+ Afterwards evaluation can be run like `cd kitti_eval && ./evaluate_object $base/label_2 ../data/kitti_object/result`


# Joint Object Pose Estimation and Shape Reconstruction in Urban Street Scenes Using 3D Shape Priors - GCPR'16

This is the code from our GCPR'16 submission: "Joint Object Pose Estimation and Shape Reconstruction in Urban Street Scenes Using 3D Shape Priors".

### Dependencies
The following libraries are required:
* VTK 7
* OpenCV
* Eigen
* CeresSolver

### Compiling
First build VIZ in `./external/viz` following the instructions given there.
Then build the code using cmake:
```
mkdir build; cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release; make -j
```

### Usage example:
The following is an example run. All needed precomputations are already given in `./data`.
```
‘./ShapePrior ../data/kitti/image_2/000046_ 10 10 2 000046 ../data/kitti/poses/000046.txt ../data/kitti/detections/000046_ ../data/kitti/disparity/000046_ ../data/kitti/calib/000046.txt ../data/kitti/planes/000046_ ../data/kitti/results/kapp3/ 1' 
```

Once the optimization has converged, a window will appear showing the input on the left and our result on the right.
You can zoom in onto the car, by pointing with the mouse at it and pressing <kbd>F</kbd>.
Press <kbd>esc</kbd> to close the window.

### Citation
If you find this code useful please cite us:
```
@inproceedings{EngelmannGCPR16_shapepriors, 
title = {Joint Object Pose Estimation and Shape Reconstruction in Urban Street Scenes Using {3D} Shape Priors},
author = {Francis Engelmann and J\"org St\"uckler and Bastian Leibe},
booktitle = {Proc. of the German Conference on Pattern Recognition (GCPR)},
year = {2016}}
```

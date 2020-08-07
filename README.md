# ORB-SLAM2

[original readme.md file](https://github.com/raulmur/ORB_SLAM2)

# 4. Monocular Examples

## TUM Dataset

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder.
```
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUMX.yaml PATH_TO_SEQUENCE_FOLDER
```

3. Information shown in terminal   
```
nipnie@nipnie:/media/nipnie/DATA/SLAMLearning/ORB_SLAM2$ ./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml /home/nipnie/data/rgbd_dataset_freiburg1_xyz

ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza.
This program comes with ABSOLUTELY NO WARRANTY;
This is free software, and you are welcome to redistribute it
under certain conditions. See LICENSE.txt.

Input sensor was set to: Monocular

Loading ORB Vocabulary. This could take a while...
Vocabulary loaded!


Camera Parameters:
- fx: 517.306
- fy: 516.469
- cx: 318.643
- cy: 255.314
- k1: 0.262383
- k2: -0.953104
- k3: 1.16331
- p1: -0.005358
- p2: 0.002628
- fps: 30
- color order: RGB (ignored if grayscale)

ORB Extractor Parameters:
- Number of Features: 1000
- Scale Levels: 8
- Scale Factor: 1.2
- Initial Fast Threshold: 20
- Minimum Fast Threshold: 7

-------
Start processing sequence ...
Images in the sequence: 798

Framebuffer with requested attributes not available. Using available framebuffer. You may see visual artifacts.New Map created with 100 points
-------

median tracking time: 0.0200165
mean tracking time: 0.0224968

Saving keyframe trajectory to KeyFrameTrajectory.txt ...

trajectory saved!
QObject::~QObject: Timers cannot be stopped from another thread
Segmentation fault (core dumped)
```

## KITTI Dataset  

1. Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php

2. Execute the following command. Change `KITTIX.yaml`by KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml for sequence 0 to 2, 3, and 4 to 12 respectively. Change `PATH_TO_DATASET_FOLDER` to the uncompressed dataset folder. Change `SEQUENCE_NUMBER` to 00, 01, 02,.., 11.
```
./Examples/Monocular/mono_kitti Vocabulary/ORBvoc.txt Examples/Monocular/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```

3. terminal Information
```
nipnie@nipnie:/media/nipnie/DATA/SLAMLearning/ORB_SLAM2$ ./Examples/Monocular/mono_kitti Vocabulary/ORBvoc.txt Examples/Monocular/KITTI03.yaml /home/nipnie/data/data_odometry_gray/sequences/03

ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza.
This program comes with ABSOLUTELY NO WARRANTY;
This is free software, and you are welcome to redistribute it
under certain conditions. See LICENSE.txt.

Input sensor was set to: Monocular

Loading ORB Vocabulary. This could take a while...
Vocabulary loaded!


Camera Parameters:
- fx: 721.538
- fy: 721.538
- cx: 609.559
- cy: 172.854
- k1: 0
- k2: 0
- p1: 0
- p2: 0
- fps: 10
- color order: RGB (ignored if grayscale)

ORB Extractor Parameters:
- Number of Features: 2000
- Scale Levels: 8
- Scale Factor: 1.2
- Initial Fast Threshold: 20
- Minimum Fast Threshold: 7

-------
Start processing sequence ...
Images in the sequence: 801

Framebuffer with requested attributes not available. Using available framebuffer. You may see visual artifacts.New Map created with 124 points
-------

median tracking time: 0.0273747
mean tracking time: 0.0320757

Saving keyframe trajectory to KeyFrameTrajectory.txt ...

trajectory saved!
QObject::~QObject: Timers cannot be stopped from another thread
Segmentation fault (core dumped)
```


# 5. Stereo Examples

## KITTI Dataset

1. Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php

2. Execute the following command. Change `KITTIX.yaml`to KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml for sequence 0 to 2, 3, and 4 to 12 respectively. Change `PATH_TO_DATASET_FOLDER` to the uncompressed dataset folder. Change `SEQUENCE_NUMBER` to 00, 01, 02,.., 11.
```
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```

3. terminal Information
```
nipnie@nipnie:/media/nipnie/DATA/SLAMLearning/ORB_SLAM2$ ./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI00-02.yaml /home/nipnie/data/data_odometry_gray/sequences/00

ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza.
This program comes with ABSOLUTELY NO WARRANTY;
This is free software, and you are welcome to redistribute it
under certain conditions. See LICENSE.txt.

Input sensor was set to: Stereo

Loading ORB Vocabulary. This could take a while...
Vocabulary loaded!


Camera Parameters:
- fx: 718.856
- fy: 718.856
- cx: 607.193
- cy: 185.216
- k1: 0
- k2: 0
- p1: 0
- p2: 0
- fps: 10
- color order: RGB (ignored if grayscale)

ORB Extractor Parameters:
- Number of Features: 2000
- Scale Levels: 8
- Scale Factor: 1.2
- Initial Fast Threshold: 20
- Minimum Fast Threshold: 7

Depth Threshold (Close/Far Points): 18.8008

-------
Start processing sequence ...
Images in the sequence: 4541

Framebuffer with requested attributes not available. Using available framebuffer. You may see visual artifacts.New map created with 851 points
Local Mapping STOP
Local Mapping RELEASE
Loop detected!
Local Mapping STOP
Local Mapping RELEASE
Starting Global Bundle Adjustment
Global Bundle Adjustment finished
Updating map ...
Local Mapping STOP
Local Mapping RELEASE
Map updated!
Loop detected!
Local Mapping STOP
Local Mapping RELEASE
Starting Global Bundle Adjustment
Global Bundle Adjustment finished
Updating map ...
Local Mapping STOP
Local Mapping RELEASE
Map updated!
Loop detected!
Local Mapping STOP
Local Mapping RELEASE
Starting Global Bundle Adjustment
Global Bundle Adjustment finished
Updating map ...
Local Mapping STOP
Local Mapping RELEASE
Map updated!
Loop detected!
Local Mapping STOP
Local Mapping RELEASE
Starting Global Bundle Adjustment
Global Bundle Adjustment finished
Updating map ...
Map updated!
-------

median tracking time: 0.0672899
mean tracking time: 0.0705476

Saving camera trajectory to CameraTrajectory.txt ...

trajectory saved!
```


# 6. RGB-D Example

## TUM Dataset

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). We already provide associations for some of the sequences in *Examples/RGB-D/associations/*. You can generate your own associations file executing:

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```

3. Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder. Change `ASSOCIATIONS_FILE` to the path to the corresponding associations file.

  ```
  ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
  ```


# 8. Processing your own sequences
You will need to create a settings file with the calibration of your camera. See the settings file provided for the TUM and KITTI datasets for monocular, stereo and RGB-D cameras. We use the calibration model of OpenCV. See the examples to learn how to create a program that makes use of the ORB-SLAM2 library and how to pass images to the SLAM system. Stereo input must be synchronized and rectified. RGB-D input must be synchronized and depth registered.

# 9. SLAM and Localization Modes
You can change between the *SLAM* and *Localization mode* using the GUI of the map viewer.

### SLAM Mode
This is the default mode. The system runs in parallal three threads: Tracking, Local Mapping and Loop Closing. The system localizes the camera, builds new map and tries to close loops.

### Localization Mode
This mode can be used when you have a good map of your working area. In this mode the Local Mapping and Loop Closing are deactivated. The system localizes the camera in the map (which is no longer updated), using relocalization if needed.

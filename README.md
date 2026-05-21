# Monochromatic Monocular SLAM with CNN/ViT Loop Closure

**Major Project Phase 1 | Dept. of CSE | Nitte Meenakshi Institute of Technology | AY 2025-26**

## Project Overview

A monocular SLAM pipeline optimised for monochromatic cameras, combining:
- Sensor-aware CLAHE preprocessing replacing naive greyscale conversion
- ResNet18 CNN-based loop closure detection
- DINO ViT-S/8 loop closure detection
- Quantitative evaluation on TUM RGB-D benchmark

---

## Results

### ATE (Absolute Trajectory Error) on TUM fr1/desk

| Configuration | ATE RMSE | Improvement |
|---|---|---|
| Baseline ORB-SLAM3 (naive greyscale) | 0.0307 m | — |
| CLAHE Preprocessing | 0.0155 m | ↓ 49.5% |

### Descriptor Latency Comparison

| Descriptor | Latency | Loop Closure Score |
|---|---|---|
| DBoW2 (ORB-SLAM3 default) | <1 ms | — |
| ResNet18 (CNN) | 1.5 ms/frame | 0.96–0.99 |
| DINO ViT-S/8 | 18.2 ms/frame | 0.90–0.93 |

---

## Repository Structure

```
├── config/
│   └── turtlebot3_waffle.yaml
├── ros_nodes/
│   ├── clahe_node.py
│   ├── loop_closure_node.py
│   ├── loop_closure_dino.py
│   └── ros_mono.cc
├── evaluation/
│   └── apply_clahe_tum.py
├── results/
│   ├── results_baseline.txt
│   ├── results_clahe.txt
│   └── ...
└── docs/
    └── setup.md
```
---

## Environment

- Ubuntu 20.04 LTS
- ROS Noetic
- Gazebo 11
- ORB-SLAM3
- OpenCV 4.4.0 (built from source)
- PyTorch 2.4.1 + CUDA 12.1
- TurtleBot3 Waffle (simulation)

---

## Setup

### 1. Install Dependencies

```bash
# OpenCV 4.4 from source
# Pangolin v0.6 from source
# ORB-SLAM3
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3 && ./build.sh

# Rebuild cv_bridge against OpenCV 4.4 (critical fix)
cd ~/cv_bridge_ws/src
git clone https://github.com/ros-perception/vision_opencv.git -b noetic
cd ~/cv_bridge_ws
catkin_make -DOpenCV_DIR=/usr/local/lib/cmake/opencv4
echo 'source ~/cv_bridge_ws/devel/setup.bash' >> ~/.bashrc

# Python dependencies
pip3 install torch torchvision timm evo
```

### 2. Launch Pipeline

```bash
# Terminal 1
roscore

# Terminal 2
export TURTLEBOT3_MODEL=waffle
roslaunch turtlebot3_gazebo turtlebot3_house.launch

# Terminal 3 — CLAHE preprocessing
python3 ros_nodes/clahe_node.py

# Terminal 4 — ORB-SLAM3
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:~/ORB_SLAM3/Examples_old/ROS
rosrun ORB_SLAM3 Mono ~/ORB_SLAM3/Vocabulary/ORBvoc.txt config/turtlebot3_waffle.yaml

# Terminal 5 — Loop closure (choose one)
python3 ros_nodes/loop_closure_node.py    # ResNet18
python3 ros_nodes/loop_closure_dino.py    # DINO ViT

# Terminal 6 — Teleop
export TURTLEBOT3_MODEL=waffle
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

### 3. TUM Evaluation

```bash
# Apply CLAHE to TUM dataset
python3 evaluation/apply_clahe_tum.py

# Run ORB-SLAM3 on baseline
cd ~/ORB_SLAM3
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt \
    Examples/Monocular/TUM1.yaml ~/rgbd_dataset_freiburg1_desk

# Evaluate ATE
evo_ape tum ~/rgbd_dataset_freiburg1_desk/groundtruth.txt \
    KeyFrameTrajectory.txt --align --correct_scale -v
```

---

## Key Engineering Fix

**cv_bridge OpenCV version conflict:**
cv_bridge (ROS system) linked against OpenCV 4.2 while ORB-SLAM3 uses 4.4.
This caused silent image corruption at the ROS/C++ boundary — images arrived
at 40Hz but were silently corrupted, causing "WAITING FOR IMAGES" indefinitely.

**Fix:** Rebuild cv_bridge from source against OpenCV 4.4.

---

## References

1. Campos et al., "ORB-SLAM3," IEEE T-RO, 2021
2. Arandjelovic et al., "NetVLAD," CVPR, 2016  
3. Caron et al., "DINO," ICCV, 2021
4. Engel et al., "DSO," IEEE T-PAMI, 2018
5. Sturm et al., "TUM RGB-D Benchmark," IROS, 2012

---

## Author

Anish Bharadwaj | 1NT23CS024 | Nitte Meenakshi Institute of Technology

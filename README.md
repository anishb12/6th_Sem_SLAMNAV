# Monochromatic Monocular SLAM with CNN/ViT Loop Closure

**Major Project Phase 1 | Dept. of CSE | Nitte Meenakshi Institute of Technology | AY 2025-26**

---

## Overview

A monocular SLAM pipeline optimised for monochromatic cameras, combining:
- Sensor-aware CLAHE preprocessing replacing naive greyscale conversion
- ResNet18 CNN-based loop closure detection (1.5ms/frame)
- DINO ViT-S/8 loop closure detection (18.2ms/frame)
- Quantitative evaluation on TUM RGB-D benchmark (fr1/desk, fr2/xyz)

**Research Contribution:** First systematic evaluation of monochromatic-specific CLAHE preprocessing combined with CNN vs ViT loop closure descriptors for monocular SLAM.

---

## Results

### ATE (Absolute Trajectory Error)

| Sequence | Baseline RMSE | CLAHE RMSE | Improvement |
|---|---|---|---|
| TUM fr1/desk | 0.0307 m | 0.0155 m | ↓ 49.5% |
| TUM fr2/xyz | 0.0027 m | 0.0031 m | marginal |

### RPE (Relative Pose Error)

| Sequence | Baseline RMSE | CLAHE RMSE |
|---|---|---|
| TUM fr1/desk | 0.0142 m | 0.0144 m |
| TUM fr2/xyz | 0.0055 m | 0.0054 m |

### Descriptor Comparison

| Descriptor | Latency | Loop Closure Score | Edge Viable |
|---|---|---|---|
| DBoW2 (ORB-SLAM3 default) | <1 ms | — | [YES] |
| ResNet18 (CNN) | 1.5 ms/frame | 0.96–0.99 | [YES] |
| DINO ViT-S/8 | 18.2 ms/frame | 0.90–0.93 | [MARGINAL] |

### Key Findings

- CLAHE reduces ATE by **49.5%** on challenging indoor scenes (fr1/desk)
- CLAHE has negligible effect on well-lit, high-texture scenes (fr2/xyz)
- ResNet18 is clearly viable for real-time edge deployment at 1.5ms/frame
- DINO ViT shows slightly lower similarity scores but stronger semantic understanding
- RPE is virtually unchanged — CLAHE improves global consistency, not local accuracy

---

## Repository Structure

```
├── config/
│   └── turtlebot3_waffle.yaml      # Camera calibration (640x480, fx=403)
├── ros_nodes/
│   ├── clahe_node.py               # CLAHE preprocessing node
│   ├── loop_closure_node.py        # ResNet18 loop closure → /loop_closure/resnet18
│   ├── loop_closure_dino.py        # DINO ViT loop closure → /loop_closure/dino
│   └── ros_mono.cc                 # Modified ORB-SLAM3 ROS wrapper
├── scripts/
│   ├── launch.sh                   # Launch with ResNet18
│   ├── launch_dino.sh              # Launch with DINO ViT
│   └── launch_comparison.sh        # Launch both side by side
├── evaluation/
│   └── apply_clahe_tum.py          # Apply CLAHE to TUM dataset
├── results/
│   ├── results_baseline.txt        # fr1/desk baseline trajectory
│   ├── results_clahe.txt           # fr1/desk CLAHE trajectory
│   ├── results_fr2_baseline.txt    # fr2/xyz baseline trajectory
│   ├── results_fr2_clahe.txt       # fr2/xyz CLAHE trajectory
│   └── ...                         # Multiple run results
└── docs/
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
- evo evaluation library

---

## Quick Start

### Install Dependencies

```bash
# OpenCV 4.4 from source (required)
# Pangolin v0.6 from source (required)

# ORB-SLAM3
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3 && ./build.sh

# Rebuild cv_bridge against OpenCV 4.4 — CRITICAL FIX
cd ~/cv_bridge_ws/src
git clone https://github.com/ros-perception/vision_opencv.git -b noetic
cd ~/cv_bridge_ws
catkin_make -DOpenCV_DIR=/usr/local/lib/cmake/opencv4
echo 'source ~/cv_bridge_ws/devel/setup.bash' >> ~/.bashrc
source ~/.bashrc

# Python dependencies
pip3 install torch torchvision timm evo
```

### Launch Pipeline

```bash
# ResNet18 loop closure
chmod +x scripts/launch.sh
./scripts/launch.sh

# DINO ViT loop closure
chmod +x scripts/launch_dino.sh
./scripts/launch_dino.sh

# Side-by-side comparison (both running simultaneously)
chmod +x scripts/launch_comparison.sh
./scripts/launch_comparison.sh
```

### TUM Evaluation

```bash
# Download TUM sequences
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz

# Apply CLAHE preprocessing
python3 evaluation/apply_clahe_tum.py

# Run ORB-SLAM3 on baseline
cd ~/ORB_SLAM3
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt \
    Examples/Monocular/TUM1.yaml ~/rgbd_dataset_freiburg1_desk

# Evaluate ATE
evo_ape tum ~/rgbd_dataset_freiburg1_desk/groundtruth.txt \
    KeyFrameTrajectory.txt --align --correct_scale -v

# Evaluate RPE
evo_rpe tum ~/rgbd_dataset_freiburg1_desk/groundtruth.txt \
    KeyFrameTrajectory.txt --align --correct_scale -v
```

---

## Critical Engineering Fix

**cv_bridge OpenCV version conflict:**

cv_bridge (ROS system library) was linked against OpenCV 4.2 while ORB-SLAM3
uses OpenCV 4.4. This caused silent image data corruption at the ROS/C++
boundary — images arrived at 40Hz but were silently corrupted, causing
"WAITING FOR IMAGES" indefinitely despite confirmed image delivery.

**Fix:** Rebuild cv_bridge from source against OpenCV 4.4.

---

## Pipeline Architecture

```
Gazebo RGB Camera
       ↓
CLAHE Preprocessing Node    ← Novel contribution 1
       ↓
ORB-SLAM3 Monocular Tracking
       ↓
ResNet18 Loop Closure       ← Novel contribution 2a → /loop_closure/resnet18
DINO ViT Loop Closure       ← Novel contribution 2b → /loop_closure/dino
       ↓
Trajectory & Map Output
```

---

## Future Work

- Integrate loop closure candidates into ORB-SLAM3 C++ pose graph optimisation
- Deploy on NVIDIA Jetson Orin Nano for edge hardware characterisation
- Evaluate on real monochromatic camera hardware
- Extend to fr3 sequences for additional benchmark coverage

---

## References

1. Campos et al., "ORB-SLAM3," IEEE T-RO, 2021
2. Arandjelovic et al., "NetVLAD," CVPR, 2016
3. Caron et al., "DINO," ICCV, 2021
4. Engel et al., "DSO," IEEE T-PAMI, 2018
5. Sturm et al., "TUM RGB-D Benchmark," IROS, 2012
6. He et al., "ResNet," CVPR, 2016
7. Lowry et al., "Visual Place Recognition Survey," IEEE T-RO, 2016

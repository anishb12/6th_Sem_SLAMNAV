# Monochromatic Monocular SLAM with Sensor-Aware Preprocessing and CNN/ViT Loop Closure

**Major Project Phase 1 | Dept. of Computer Science & Engineering**  
**Nitte Meenakshi Institute of Technology, Bengaluru | AY 2025–26**

> **Guide:** Dr. Krishna Rao Venkatesh, Professor, Dept. of CSE, NMIT  
> **Students:** Anish Bharadwaj | 1NT23CS024 &nbsp;&nbsp; Dhanush Paturu | 1NT23CS060 &nbsp;&nbsp; Dhruva P S | 1NT23CS064

---

## What This Project Does

This project builds a **monocular SLAM system** (Simultaneous Localisation and Mapping) specifically optimised for **monochromatic cameras** — cameras that capture a single luminance channel rather than colour. Such cameras are increasingly used in industrial robots and drones because they are cheaper, sharper, and more sensitive in low light than colour cameras.

The problem is that all existing SLAM systems treat monochromatic input the same way they treat colour camera input — by applying a fixed greyscale formula designed for colour sensors. This project asks: *does applying sensor-specific preprocessing improve SLAM accuracy, and which deep learning descriptor works best for loop closure on edge hardware?*

**SLAM** means the robot builds a map of its environment and estimates its own position at the same time — using only a camera, no GPS, no depth sensor.

**Loop closure** means the robot recognises when it has returned to a previously visited location and uses that recognition to correct accumulated position drift.

---

## Research Contribution

> First systematic evaluation of monochromatic-specific CLAHE preprocessing combined with CNN (ResNet18) versus Vision Transformer (DINO ViT-S/8) loop closure descriptors for monocular SLAM — benchmarked on standard TUM RGB-D sequences.

Three specific gaps in existing literature are addressed:

1. No SLAM system applies sensor-specific preprocessing for monochromatic cameras
2. No published benchmark measures the effect of preprocessing on monocular SLAM trajectory accuracy
3. No comparison of CNN vs Vision Transformer loop closure descriptors within a real-time SLAM thread on edge hardware

---

## What Was Built

### Contribution 1 — CLAHE Preprocessing Node
A ROS node that replaces the naive RGB-to-greyscale conversion in ORB-SLAM3 with **Contrast Limited Adaptive Histogram Equalisation (CLAHE)** tuned for monochromatic sensor characteristics. CLAHE enhances local contrast in 8×8 tiles without amplifying noise — making corners and edges more visible for ORB feature detection.

### Contribution 2 — ResNet18 CNN Loop Closure Module
A Python ROS node that extracts a **512-dimensional scene descriptor** from each keyframe using a pretrained ResNet18 CNN. If the cosine similarity between two keyframe descriptors exceeds 0.85, a loop closure candidate is flagged. Achieves **1.5ms per frame** — viable for real-time edge deployment.

### Contribution 3 — DINO ViT-S/8 Loop Closure Module
An alternative loop closure module using a **DINO Vision Transformer** (ViT-S/8). The image is split into 8×8 patches and processed with global self-attention, producing a 384-dimensional [CLS] token descriptor. Achieves stronger semantic understanding than ResNet18 at the cost of higher latency (**18.2ms per frame**).

---

## Key Results

### Trajectory Accuracy on TUM RGB-D Benchmark

| Sequence | Configuration | ATE RMSE | RPE RMSE | Improvement |
|---|---|---|---|---|
| TUM fr1/desk | Baseline ORB-SLAM3 | 0.0307 m | 0.0142 m | — |
| TUM fr1/desk | **CLAHE Variant** | **0.0155 m** | 0.0144 m | **↓ 49.5%** |
| TUM fr2/xyz | Baseline ORB-SLAM3 | 0.0027 m | 0.0055 m | — |
| TUM fr2/xyz | CLAHE Variant | 0.0031 m | 0.0054 m | Marginal |

**ATE** (Absolute Trajectory Error) measures overall path drift from ground truth.  
**RPE** (Relative Pose Error) measures local step-by-step accuracy.

**Finding:** CLAHE reduces ATE by 49.5% on challenging indoor scenes (fr1/desk). Negligible effect on well-lit, high-texture scenes (fr2/xyz) where the baseline already performs near-perfectly.

### Descriptor Latency Comparison

| Descriptor | Latency | Loop Closure Score | Edge Viable |
|---|---|---|---|
| DBoW2 (ORB-SLAM3 default) | <1 ms | — | Yes |
| ResNet18 (CNN) | 1.5 ms/frame | 0.96–0.99 | Yes |
| DINO ViT-S/8 | 18.2 ms/frame | 0.90–0.93 | Marginal |

---

## System Architecture

```
Gazebo RGB Camera
       |
       | /camera/rgb/image_raw
       v
CLAHE Preprocessing Node        <-- Novel contribution 1
       |
       | /camera/image_raw (enhanced greyscale)
       v
ORB-SLAM3 Monocular Tracking
       |
       |-- Keyframes + Pose
       |
       v
ResNet18 Loop Closure Module    <-- Novel contribution 2a
       | /loop_closure/resnet18

DINO ViT Loop Closure Module    <-- Novel contribution 2b
       | /loop_closure/dino
```

All components communicate via ROS publish-subscribe topics. The two loop closure modules run in parallel — switch between them using the launch script argument.

---

## Documents

- `docs/SLAM_Report.pdf` — Full synopsis report with results, methodology, literature review, and 10 figures
- `docs/SLAM_Presentation.pptx` —  Presentation

---

## Execution Environment

### Hardware
- CPU: Intel i7 12th Gen (tested environment)
- GPU: NVIDIA RTX 3060 (CUDA 12.1)
- RAM: 16 GB

> All results were obtained on this hardware configuration.
> Performance on other hardware may vary, particularly for 
> deep learning inference latency.

### Software

| Component | Version |
|---|---|
| OS | Ubuntu 20.04 LTS |
| ROS | Noetic (desktop-full) |
| Gazebo | 11 |
| Python | 3.8 |
| OpenCV | 4.4.0 (built from source) |
| PyTorch | 2.4.1 + CUDA 12.1 |
| ORB-SLAM3 | UZ-SLAMLab/ORB_SLAM3 (latest) |
| Pangolin | v0.6 |
| evo | 1.31.1 |

### Python Dependencies
```bash
pip3 install torch torchvision timm evo
```

---

## Installation and Setup

### Step 1 — Install ROS Noetic
```bash
sudo apt install ros-noetic-desktop-full
sudo apt install ros-noetic-turtlebot3 ros-noetic-turtlebot3-simulations
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 2 — Build OpenCV 4.4 from Source
The system OpenCV (4.2) is incompatible with ORB-SLAM3. Build 4.4 from source:
```bash
cd ~
git clone https://github.com/opencv/opencv.git
cd opencv && git checkout 4.4.0
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_EXAMPLES=OFF ..
make -j$(nproc)
sudo make install
```

### Step 3 — Build Pangolin v0.6
```bash
cd ~
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin && git checkout v0.6
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

### Step 4 — Build ORB-SLAM3
```bash
cd ~
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3
chmod +x build.sh
./build.sh
```

### Step 5 — Rebuild cv_bridge Against OpenCV 4.4 (CRITICAL)

> **Why this is required:** The ROS system cv_bridge library is linked against OpenCV 4.2. When it passes image data to ORB-SLAM3 (which uses OpenCV 4.4), the internal cv::Mat data structures are incompatible — images are silently corrupted at the ROS/C++ boundary. This causes ORB-SLAM3 to print "WAITING FOR IMAGES" indefinitely despite images arriving correctly at 40Hz. Rebuilding cv_bridge from source against OpenCV 4.4 resolves this completely.

```bash
mkdir -p ~/cv_bridge_ws/src && cd ~/cv_bridge_ws/src
git clone https://github.com/ros-perception/vision_opencv.git -b noetic
cd ~/cv_bridge_ws
catkin_make -DOpenCV_DIR=/usr/local/lib/cmake/opencv4
echo "source ~/cv_bridge_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 6 — Install Python Dependencies
```bash
pip3 install torch torchvision timm evo
```

### Step 7 — Clone This Repository
```bash
cd ~
git clone https://github.com/anishb12/6th_Sem_SLAMNAV.git mono-slam-cnn-loop-closure
```

### Step 8 — Build the ROS Workspace
```bash
mkdir -p ~/catkin_ws/src/clahe_preprocessor/src
cp ~/mono-slam-cnn-loop-closure/ros_nodes/*.py ~/catkin_ws/src/clahe_preprocessor/src/
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

---

## Running the Pipeline

### Launch Everything with One Command

```bash
chmod +x ~/mono-slam-cnn-loop-closure/scripts/launch.sh

# ResNet18 loop closure (default)
~/mono-slam-cnn-loop-closure/scripts/launch.sh

# DINO ViT-S/8 loop closure
~/mono-slam-cnn-loop-closure/scripts/launch.sh dino
```

This opens 6 terminals automatically:
1. roscore
2. Gazebo house world with TurtleBot3 Waffle
3. CLAHE preprocessing node
4. ORB-SLAM3 monocular tracking
5. Loop closure module (ResNet18 or DINO ViT)
6. Teleop keyboard control (W/A/S/D keys to drive)

### Manual Launch (fallback)

```bash
# Terminal 1
roscore

# Terminal 2
export TURTLEBOT3_MODEL=waffle
roslaunch turtlebot3_gazebo turtlebot3_house.launch

# Terminal 3
python3 ~/mono-slam-cnn-loop-closure/ros_nodes/clahe_node.py

# Terminal 4
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:~/ORB_SLAM3/Examples_old/ROS
rosrun ORB_SLAM3 Mono ~/ORB_SLAM3/Vocabulary/ORBvoc.txt \
    ~/mono-slam-cnn-loop-closure/config/turtlebot3_waffle.yaml

# Terminal 5 — choose one
python3 ~/mono-slam-cnn-loop-closure/ros_nodes/loop_closure_node.py   # ResNet18
python3 ~/mono-slam-cnn-loop-closure/ros_nodes/loop_closure_dino.py   # DINO ViT

# Terminal 6
export TURTLEBOT3_MODEL=waffle
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

---

## TUM Dataset Evaluation

To reproduce the ATE/RPE results:

```bash
# Download TUM fr1/desk sequence
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar -xzf rgbd_dataset_freiburg1_desk.tgz

# Apply CLAHE preprocessing
python3 ~/mono-slam-cnn-loop-closure/evaluation/apply_clahe_tum.py

# Run ORB-SLAM3 on baseline
cd ~/ORB_SLAM3
./Examples/Monocular/mono_tum \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular/TUM1.yaml \
    ~/rgbd_dataset_freiburg1_desk

# Evaluate ATE
evo_ape tum \
    ~/rgbd_dataset_freiburg1_desk/groundtruth.txt \
    KeyFrameTrajectory.txt \
    --align --correct_scale -v

# Evaluate RPE
evo_rpe tum \
    ~/rgbd_dataset_freiburg1_desk/groundtruth.txt \
    KeyFrameTrajectory.txt \
    --align --correct_scale -v
```

Repeat with `~/rgbd_dataset_freiburg1_desk_clahe` for CLAHE results.  
Use `TUM2.yaml` and `rgbd_dataset_freiburg2_xyz` for the fr2/xyz sequence.

Pre-computed trajectory files for all runs are in `results/`.

---

## Repository Structure

```
├── config/
│   └── turtlebot3_waffle.yaml      Camera calibration (640x480, fx=403)
├── ros_nodes/
│   ├── clahe_node.py               CLAHE preprocessing node
│   ├── loop_closure_node.py        ResNet18 loop closure (512-d)
│   ├── loop_closure_dino.py        DINO ViT-S/8 loop closure (384-d)
│   └── ros_mono.cc                 Modified ORB-SLAM3 ROS wrapper
├── scripts/
│   └── launch.sh                   ./launch.sh or ./launch.sh dino
├── evaluation/
│   └── apply_clahe_tum.py          Applies CLAHE to TUM sequences
├── results/
│   ├── results_baseline.txt        fr1/desk baseline trajectory
│   ├── results_clahe.txt           fr1/desk CLAHE trajectory
│   ├── results_fr2_baseline.txt    fr2/xyz baseline trajectory
│   └── results_fr2_clahe.txt       fr2/xyz CLAHE trajectory
├── figures/                        All result charts and diagrams
└── docs/
    ├── SLAM_Report_With_Figures.docx
    └── SLAM_Final_Presentation_v2.pptx
```

---

## Limitations

- Monochromatic input is simulated by applying CLAHE to colour camera footage — a real monochromatic sensor has not been used
- The ResNet18 and DINO ViT modules detect loop closure candidates and publish them via ROS topics, but are not yet integrated into ORB-SLAM3's C++ pose graph optimisation. The 49.5% ATE improvement is entirely from CLAHE preprocessing
- 2 valid runs per configuration — 5+ runs would give stronger statistical significance

---

## Future Work

- Integrate loop closure candidates into ORB-SLAM3's LoopClosing.cc pose graph thread
- Deploy on NVIDIA Jetson Orin Nano — measure accuracy vs latency tradeoff
- Evaluate on real monochromatic camera hardware
- Extend benchmark to TUM fr3 sequences

---

## References

1. C. Campos et al., "ORB-SLAM3," IEEE T-RO, vol. 37, no. 6, 2021
2. R. Arandjelovic et al., "NetVLAD," CVPR, 2016
3. M. Caron et al., "DINO," ICCV, 2021
4. J. Engel et al., "DSO," IEEE T-PAMI, vol. 40, no. 3, 2018
5. N. Keetha et al., "AnyLoc," IEEE RA-L, 2024
6. J. Sturm et al., "TUM RGB-D Benchmark," IROS, 2012
7. K. He et al., "ResNet," CVPR, 2016
8. S. Lowry et al., "Visual Place Recognition Survey," IEEE T-RO, vol. 32, no. 1, 2016

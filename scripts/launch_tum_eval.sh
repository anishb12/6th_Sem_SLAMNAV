#!/bin/bash
# Launch pipeline for TUM benchmark evaluation with pose-graph loop closure injection

MODE=${1:-resnet18}
DATASET=${2:-$HOME/rgbd_dataset_freiburg1_desk}

echo "Starting TUM evaluation pipeline: $MODE on $DATASET"

source /opt/ros/noetic/setup.bash
source ~/cv_bridge_ws/devel/setup.bash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:~/ORB_SLAM3/Examples_old/ROS

# Terminal 1 — roscore
gnome-terminal --title="roscore" -- bash -c "roscore; exec bash" &
sleep 3

# Terminal 2 — CLAHE node
gnome-terminal --title="CLAHE Node" -- bash -c "
    source /opt/ros/noetic/setup.bash
    source ~/cv_bridge_ws/devel/setup.bash
    python3 ~/mono-slam-cnn-loop-closure/ros_nodes/clahe_node.py;
    exec bash" &
sleep 3

# Terminal 3 — ORB-SLAM3 (with TUM calibration, not turtlebot3)
gnome-terminal --title="ORB-SLAM3" -- bash -c "
    source /opt/ros/noetic/setup.bash
    source ~/cv_bridge_ws/devel/setup.bash
    export ROS_PACKAGE_PATH=\$ROS_PACKAGE_PATH:~/ORB_SLAM3/Examples_old/ROS
    rosrun ORB_SLAM3 Mono ~/ORB_SLAM3/Vocabulary/ORBvoc.txt \
        ~/ORB_SLAM3/Examples/Monocular/TUM1.yaml;
    exec bash" &
sleep 5

# Terminal 4 — Loop closure module
if [ "$MODE" = "dino" ]; then
    gnome-terminal --title="Loop Closure - DINO ViT" -- bash -c "
        source /opt/ros/noetic/setup.bash
        source ~/cv_bridge_ws/devel/setup.bash
        python3 ~/mono-slam-cnn-loop-closure/ros_nodes/loop_closure_dino.py;
        exec bash" &
elif [ "$MODE" = "efficientnet" ]; then
    gnome-terminal --title="Loop Closure - EfficientNet-B0" -- bash -c "
        source /opt/ros/noetic/setup.bash
        source ~/cv_bridge_ws/devel/setup.bash
        python3 ~/mono-slam-cnn-loop-closure/ros_nodes/loop_closure_efficientnet.py;
        exec bash" &
else
    gnome-terminal --title="Loop Closure - ResNet18" -- bash -c "
        source /opt/ros/noetic/setup.bash
        source ~/cv_bridge_ws/devel/setup.bash
        python3 ~/mono-slam-cnn-loop-closure/ros_nodes/loop_closure_node.py;
        exec bash" &
fi
sleep 2

# Terminal 5 — TUM image publisher (drives the whole pipeline)
gnome-terminal --title="TUM Publisher" -- bash -c "
    source /opt/ros/noetic/setup.bash
    python3 ~/mono-slam-cnn-loop-closure/evaluation/tum_publisher.py $DATASET;
    exec bash" &

echo "All nodes launched. TUM playback will drive the pipeline automatically."
echo "Usage: ./launch_tum_eval.sh [resnet18|dino|efficientnet] [dataset_path]"

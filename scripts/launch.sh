#!/bin/bash

# Monochromatic Monocular SLAM — Full Pipeline Launch Script
echo "Starting full SLAM pipeline..."

# Source ROS
source /opt/ros/noetic/setup.bash
source ~/cv_bridge_ws/devel/setup.bash
export TURTLEBOT3_MODEL=waffle
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:~/ORB_SLAM3/Examples_old/ROS

# Terminal 1 — roscore
gnome-terminal --title="roscore" -- bash -c "roscore; exec bash" &
sleep 3

# Terminal 2 — Gazebo
gnome-terminal --title="Gazebo" -- bash -c "
    export TURTLEBOT3_MODEL=waffle
    roslaunch turtlebot3_gazebo turtlebot3_house.launch;
    exec bash" &
sleep 8

# Terminal 3 — CLAHE node
gnome-terminal --title="CLAHE Node" -- bash -c "
    source /opt/ros/noetic/setup.bash
    source ~/cv_bridge_ws/devel/setup.bash
    python3 ~/mono-slam-cnn-loop-closure/ros_nodes/clahe_node.py;
    exec bash" &
sleep 3

# Terminal 4 — ORB-SLAM3
gnome-terminal --title="ORB-SLAM3" -- bash -c "
    source /opt/ros/noetic/setup.bash
    source ~/cv_bridge_ws/devel/setup.bash
    export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:~/ORB_SLAM3/Examples_old/ROS
    rosrun ORB_SLAM3 Mono ~/ORB_SLAM3/Vocabulary/ORBvoc.txt \
        ~/mono-slam-cnn-loop-closure/config/turtlebot3_waffle.yaml;
    exec bash" &
sleep 5

# Terminal 5 — Loop closure (ResNet18 by default)
gnome-terminal --title="Loop Closure" -- bash -c "
    source /opt/ros/noetic/setup.bash
    source ~/cv_bridge_ws/devel/setup.bash
    python3 ~/mono-slam-cnn-loop-closure/ros_nodes/loop_closure_node.py;
    exec bash" &
sleep 2

# Terminal 6 — Teleop
gnome-terminal --title="Teleop" -- bash -c "
    source /opt/ros/noetic/setup.bash
    export TURTLEBOT3_MODEL=waffle
    roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch;
    exec bash" &

echo "All nodes launched. Use the Teleop terminal to drive the robot."
echo "To use DINO ViT instead of ResNet18, edit this script and change loop_closure_node.py to loop_closure_dino.py"

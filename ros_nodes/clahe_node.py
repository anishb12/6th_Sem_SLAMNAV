#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
pub = None

def callback(msg):
    # Convert ROS image to OpenCV
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
    # Resize to working resolution
    img = cv2.resize(img, (640, 480))
    
    # Convert to greyscale — simulating monochromatic sensor
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE — sensor-aware contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(grey)
    
    # Convert back to ROS and publish
    out_msg = bridge.cv2_to_imgmsg(enhanced, encoding='mono8')
    out_msg.header = msg.header
    pub.publish(out_msg)

def main():
    global pub
    rospy.init_node('clahe_preprocessor')
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)
    rospy.Subscriber('/camera/rgb/image_raw', Image, callback)
    rospy.loginfo("CLAHE preprocessing node started")
    rospy.spin()

if __name__ == '__main__':
    main()

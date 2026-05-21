#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

bridge = CvBridge()

# Load DINO ViT-S/8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rospy.loginfo(f"Using device: {device}")

model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
model.eval()
model.to(device)

# ImageNet normalisation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Descriptor database
descriptor_db = []
keyframe_count = 0
SIMILARITY_THRESHOLD = 0.85
pub = None

def extract_descriptor(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze().cpu().numpy()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def callback(msg):
    global keyframe_count

    keyframe_count += 1
    if keyframe_count % 5 != 0:
        return

    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    descriptor = extract_descriptor(img)

    best_score = 0.0
    best_id = -1
    search_db = descriptor_db[:-10] if len(descriptor_db) > 10 else []
    for kf_id, db_desc in search_db:
        score = cosine_similarity(descriptor, db_desc)
        if score > best_score:
            best_score = score
            best_id = kf_id

    if best_score > SIMILARITY_THRESHOLD:
        msg_out = f"LOOP CLOSURE: KF {keyframe_count} matches KF {best_id} | score={best_score:.3f}"
        rospy.logwarn(msg_out)
        pub.publish(msg_out)
    else:
        rospy.loginfo(f"KF {keyframe_count} | best_match={best_id} | score={best_score:.3f}")

    descriptor_db.append((keyframe_count, descriptor))

def main():
    global pub
    rospy.init_node('loop_closure_dino')
    pub = rospy.Publisher('/loop_closure/candidates', String, queue_size=10)
    rospy.Subscriber('/camera/image_raw', Image, callback)
    rospy.loginfo("DINO ViT loop closure node started")
    rospy.spin()

if __name__ == '__main__':
    main()

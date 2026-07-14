#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import timm
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import time

# Load EfficientNet-B0 — num_classes=0 gives 1280-d global pool output directly
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

bridge = CvBridge()
descriptor_db = []
kf_counter = 0
THRESHOLD = 0.85
SKIP_RECENT = 10

def extract_descriptor(frame_grey):
    frame_rgb = cv2.cvtColor(frame_grey, cv2.COLOR_GRAY2RGB)
    tensor = transform(frame_rgb).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        desc = model(tensor).squeeze()  # Shape: [1280]
    return F.normalize(desc, dim=0)

def callback(msg):
    global kf_counter
    kf_counter += 1
    if kf_counter % 5 != 0:
        return

    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

    t0 = time.time()
    desc = extract_descriptor(frame)
    latency = (time.time() - t0) * 1000  # ms

    best_score, best_id = 0, -1
    for (kid, kd) in descriptor_db[:-SKIP_RECENT] if len(descriptor_db) > SKIP_RECENT else []:
        score = torch.dot(desc, kd).item()
        if score > best_score:
            best_score, best_id = score, kid

    descriptor_db.append((kf_counter, desc))

    if best_score > THRESHOLD:
        msg_out = (f"LOOP CLOSURE: KF {kf_counter} matches KF {best_id} "
                   f"| score={best_score:.3f} | latency={latency:.2f}ms")
        pub.publish(msg_out)
        rospy.logwarn(msg_out)
    else:
        rospy.loginfo(f"KF {kf_counter} processed | best_score={best_score:.3f} "
                      f"| latency={latency:.2f}ms | db_size={len(descriptor_db)}")

rospy.init_node('loop_closure_efficientnet')
sub = rospy.Subscriber('/camera/image_raw', Image, callback)
pub = rospy.Publisher('/loop_closure/efficientnet', String, queue_size=10)
rospy.loginfo("EfficientNet-B0 loop closure node started — descriptor dim: 1280")
rospy.spin()

import cv2
import os
import shutil

src = os.path.expanduser("~/rgbd_dataset_freiburg1_desk/rgb")
dst = os.path.expanduser("~/rgbd_dataset_freiburg1_desk_clahe/rgb")
os.makedirs(dst, exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

files = sorted(os.listdir(src))
for i, f in enumerate(files):
    img = cv2.imread(os.path.join(src, f))
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(grey)
    # Save as 3-channel so ORB-SLAM3 accepts it
    out = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(dst, f), out)
    if i % 100 == 0:
        print(f"Processed {i}/{len(files)}")

# Copy metadata files
for f in ["rgb.txt", "groundtruth.txt", "depth.txt"]:
    shutil.copy(
        os.path.expanduser(f"~/rgbd_dataset_freiburg1_desk/{f}"),
        os.path.expanduser(f"~/rgbd_dataset_freiburg1_desk_clahe/{f}")
    )

print("Done — CLAHE dataset ready")

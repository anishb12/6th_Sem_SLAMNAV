import cv2
import numpy as np
import matplotlib.pyplot as plt

orb = cv2.ORB_create(2000) # ORB detector
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # Matcher
cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

traj = np.zeros((600, 600, 3), dtype=np.uint8) # Camera pose (x, z in 2D)
x, z = 300, 300

plt.ion()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    if des is not None and prev_des is not None:
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 10:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            E, mask = cv2.findEssentialMat(
                pts1, pts2,
                focal=1.0, pp=(0.,0.),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2)
                dx = int(t[0][0] * 5)
                dz = int(t[2][0] * 5)
                x += dx
                z += dz
                cv2.circle(traj, (x,z), 2, (0,255,0), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Trajectory", traj)
    prev_gray = gray
    prev_kp, prev_des = kp, des
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
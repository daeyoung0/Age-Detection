"""Raw IP-camera preview with NO model inference.

Use this to isolate whether a black screen is a stream/URL problem or just
slow CPU inference in the full app. If video shows here but the main app is
black, the cause is inference latency, not the camera.

    python preview_camera.py http://192.168.219.101:8050/video
"""
import sys
import time

import cv2
import numpy as np

url = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.219.101:8050/video"
print(f"[preview] opening {url}")
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("[preview] FAILED to open stream")
    sys.exit(1)

fps_t = time.time()
n = 0
while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[preview] read failed")
        continue
    n += 1
    if n % 30 == 0:
        dt = time.time() - fps_t
        print(f"[preview] {n} frames | ~{30/dt:.1f} fps | mean_brightness={float(np.asarray(frame).mean()):.1f}")
        fps_t = time.time()
    cv2.imshow("IP Camera Preview (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

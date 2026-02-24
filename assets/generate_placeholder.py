"""
Run once to create a synthetic cockpit placeholder image if you don't have
a real cockpit photograph:

    python assets/generate_placeholder.py
"""
import os
import sys

try:
    import numpy as np
    import cv2
except ImportError:
    print("numpy/opencv not installed yet – run 'pip install numpy opencv-python' first.")
    sys.exit(1)

_OUT = os.path.join(os.path.dirname(__file__), "cockpit.jpg")

W, H = 1280, 720
img = np.zeros((H, W, 3), dtype=np.uint8)

# Dark cockpit background
img[:, :] = (30, 28, 22)

# Instrument panel region (bottom third)
panel_y = int(H * 0.55)
cv2.rectangle(img, (0, panel_y), (W, H), (50, 48, 40), -1)

# Instrument circles
instruments = [
    (200, panel_y + 80, 60),
    (400, panel_y + 80, 60),
    (600, panel_y + 80, 60),
    (800, panel_y + 80, 60),
    (1000, panel_y + 80, 60),
    (300, panel_y + 200, 50),
    (500, panel_y + 200, 55),
    (700, panel_y + 200, 55),
    (900, panel_y + 200, 50),
]
for cx, cy, r in instruments:
    cv2.circle(img, (cx, cy), r, (80, 100, 90), -1)
    cv2.circle(img, (cx, cy), r, (120, 150, 130), 2)
    cv2.circle(img, (cx, cy), 4, (200, 220, 210), -1)

# Horizon line (artificial horizon in centre)
ah_cx, ah_cy, ah_r = W // 2, int(H * 0.35), 90
cv2.circle(img, (ah_cx, ah_cy), ah_r, (60, 80, 70), -1)
cv2.circle(img, (ah_cx, ah_cy), ah_r, (140, 170, 150), 3)
# Sky/ground split
cv2.rectangle(img, (ah_cx - ah_r, ah_cy - ah_r), (ah_cx + ah_r, ah_cy), (80, 100, 150), -1)
cv2.circle(img, (ah_cx, ah_cy), ah_r, (140, 170, 150), 3)

# Windshield frame
cv2.rectangle(img, (0, 0), (W, panel_y - 10), (20, 18, 14), 20)

# Label
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "COCKPIT VIEW  (placeholder)", (W // 2 - 200, H - 20),
            font, 0.7, (100, 100, 80), 1, cv2.LINE_AA)

cv2.imwrite(_OUT, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(f"Placeholder cockpit image written to: {_OUT}")

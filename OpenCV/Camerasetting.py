import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        break
    print(f"Camera {i}: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    cap.release()
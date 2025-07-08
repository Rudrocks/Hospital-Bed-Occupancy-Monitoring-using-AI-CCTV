import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use yolov8n.pt (nano version)

# Capture video from webcam (or replace '0' with CCTV RTSP link)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    bed_count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 56:  # Class 56 in COCO is 'chair' (used as proxy for bed)
                bed_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Beds Detected (proxy): {bed_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Hospital Bed Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
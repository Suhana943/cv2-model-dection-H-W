import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 general COCO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Average diameter in cm (optional for rough real-world size)
AVG_DIAMETER_CM = 10  # e.g., tennis ball ~7 cm, plate ~20 cm → pick an average

# List of round objects in YOLO COCO model
ROUND_CLASSES = ["sports ball", "cup", "bowl", "clock"]

# You can add "bottle cap" or "plate" if you have a custom model

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detect
    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0].item())
            label = model.names[cls]
            conf = box.conf[0].item()

            # Only process round objects
            if label in ROUND_CLASSES:
                # Get box region
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]

                # Convert to gray
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)

                # Hough Circle Transform
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                                           minDist=20,
                                           param1=50, param2=30,
                                           minRadius=10, maxRadius=300)

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :1]:  # Just the first circle per ROI
                        center = (i[0] + x1, i[1] + y1)  # Adjust to full frame
                        radius = i[2]

                        diameter_pixels = radius * 2
                        pixels_per_cm = diameter_pixels / AVG_DIAMETER_CM
                        radius_cm = radius / pixels_per_cm

                        # Draw circle & center
                        cv2.circle(frame, center, radius, (0, 255, 0), 2)
                        cv2.circle(frame, center, 2, (0, 0, 255), 3)

                        cv2.putText(frame, f"{label} R: {radius}px ({radius_cm:.1f}cm)",
                                    (center[0] - 40, center[1] - radius - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Draw YOLO box for reference
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Round Object Radius Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

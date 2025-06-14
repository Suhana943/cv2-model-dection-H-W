import cv2
from ultralytics import YOLO

# ✅ Load YOLOv8 general detection model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt for more accuracy

# ✅ Open webcam ONCE
cap = cv2.VideoCapture(0)

# ✅ Average length for approximate cm (adjust as you like)
AVG_REF_LENGTH_CM = 15  # E.g., assume longest side ~ 15cm for scaling

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Run YOLO detection
    results = model(frame)

    # ✅ Loop over detected boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0].item())  # class ID
            conf = box.conf[0].item()     # confidence
            label = model.names[cls]      # class name

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width_pixels = abs(x2 - x1)
            height_pixels = abs(y2 - y1)

            # Optional: approximate size in cm
            longer_side_pixels = max(width_pixels, height_pixels)
            pixels_per_cm = longer_side_pixels / AVG_REF_LENGTH_CM
            width_cm = width_pixels / pixels_per_cm
            height_cm = height_pixels / pixels_per_cm

            # ✅ Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ✅ Label: class name + confidence
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            # ✅ Width & Height (pixels and cm)
            cv2.putText(frame, f"W: {width_pixels}px ({width_cm:.1f}cm)",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)
            cv2.putText(frame, f"H: {height_pixels}px ({height_cm:.1f}cm)",
                        (x1, y2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)

    # ✅ Show result
    cv2.imshow("YOLOv8 ALL Objects + Box + Size", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release webcam and windows
cap.release()
cv2.destroyAllWindows()

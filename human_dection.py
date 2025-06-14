import cv2
from ultralytics import YOLO

# ✅ Load general YOLOv8 modelhuman_dection
model = YOLO('yolov8n.pt')  # COCO model with 80 classes

# ✅ Open webcam
cap = cv2.VideoCapture(0)

# OPTIONAL: use average phone length (e.g., 15 cm) for approximate scaling
AVG_PHONE_LENGTH_CM = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Run YOLOv8 detection
    results = model.predict(frame, save=False, imgsz=640, conf=0.3)

    # ✅ Loop through detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0].item())   # class index
            conf = box.conf[0].item()
            label = model.names[cls]

            if label == "cell phone":
                # Box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width_pixels = abs(x2 - x1)
                height_pixels = abs(y2 - y1)

                # (Optional) Convert to cm using an approximate scale
                # Here we just assume the longer side is ~15 cm
                longer_side_pixels = max(width_pixels, height_pixels)
                pixels_per_cm = longer_side_pixels / AVG_PHONE_LENGTH_CM
                width_cm = width_pixels / pixels_per_cm
                height_cm = height_pixels / pixels_per_cm

                # ✅ Draw box & labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"W: {width_pixels}px ({width_cm:.1f}cm)", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, f"H: {height_pixels}px ({height_cm:.1f}cm)", (x1, y2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # ✅ Show result
    cv2.imshow("YOLO Phone Detection + Size", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    
# ✅ Load YOLOv8 general object detection model (COCO)
model = YOLO('yolov8n.pt')  # use yolov8n.pt, yolov8s.pt, etc.

# ✅ Open webcam
cap = cv2.VideoCapture(0)

# Average phone length in cm (optional for approximate scaling)
AVG_PHONE_LENGTH_CM = 15  # average smartphone length in cm

while True:
    ret, frame = cap.read()
    if not ret:
       break

    # ✅ Run YOLOv8 detection
    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0].item())  # class index
            conf = box.conf[0].item()
            label = model.names[cls]

            if label == "cell phone":
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width_pixels = abs(x2 - x1)
                height_pixels = abs(y2 - y1)

                # (Optional) estimate cm using approximate scale
                longer_side_pixels = max(width_pixels, height_pixels)
                pixels_per_cm = longer_side_pixels / AVG_PHONE_LENGTH_CM
                width_cm = width_pixels / pixels_per_cm
                height_cm = height_pixels / pixels_per_cm

                # ✅ Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ✅ Draw label and confidence
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

                # ✅ Draw size info below box
                cv2.putText(frame, f"W: {width_pixels}px ({width_cm:.1f}cm)",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 0, 0), 2)

                cv2.putText(frame, f"H: {height_pixels}px ({height_cm:.1f}cm)",
                            (x1, y2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 0, 0), 2)

    # ✅ Display the result
    cv2.imshow("YOLOv8 Phone Detection + Size + Boxes", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Average reference length for approximate scale (optional)
# For general object detection, you can skip this or use any average for rough scale
AVG_REF_LENGTH_CM = 15  # adjust as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Run YOLOv8 detection
    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0].item())  # class index
            conf = box.conf[0].item()
            label = model.names[cls]

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width_pixels = abs(x2 - x1)
            height_pixels = abs(y2 - y1)

            # Optional: approximate cm using longest side as rough scale
            longer_side_pixels = max(width_pixels, height_pixels)
            pixels_per_cm = longer_side_pixels / AVG_REF_LENGTH_CM
            width_cm = width_pixels / pixels_per_cm
            height_cm = height_pixels / pixels_per_cm

            # ✅ Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ✅ Label with object name & confidence
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            # ✅ Display width & height below box
            cv2.putText(frame, f"W: {width_pixels}px ({width_cm:.1f}cm)",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2)

            cv2.putText(frame, f"H: {height_pixels}px ({height_cm:.1f}cm)",
                        (x1, y2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2)

    # ✅ Show frame
    cv2.imshow("YOLOv8 ALL Objects + Box Size", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

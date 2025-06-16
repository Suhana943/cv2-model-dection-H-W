from ultralytics import YOLO
import cv2

# ✅ Load YOLOv8 model (COCO pre-trained)
model = YOLO("yolov8n.pt")  # lightweight nano model

# ✅ Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # ✅ Run detection
    results = model(frame, stream=True)

    # ✅ Loop through detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])  # class ID
            conf = float(box.conf[0])  # confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box

            # ✅ COCO class ID for microwave = 82
            if cls == 82:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Microwave: {conf:.2f}", 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                            (0, 255, 0), 2)

    # ✅ Show frame
    cv2.imshow("Microwave Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

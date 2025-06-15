import cv2
import mediapipe as mp

# === 1) Setup ===
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection()

# === 2) Webcam ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(rgb)

    if result.detections:
        for detection in result.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                           int(bboxC.width * w), int(bboxC.height * h)

            # === Draw Face Bounding Box ===
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            # === Draw a Card ===
            card_x1, card_y1 = x + bw + 20, y
            card_x2, card_y2 = card_x1 + 200, y + 100
            cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (255, 255, 255), -1)
            cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (0, 0, 0), 2)

            # === Add Info on Card ===
            cv2.putText(frame, "Name: Suhana", (card_x1 + 10, card_y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, "Mood: Happy 😊", (card_x1 + 10, card_y1 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Face Card", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

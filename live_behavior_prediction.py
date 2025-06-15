import cv2
import mediapipe as mp
import numpy as np
import joblib
import math

# === 1) Load your trained SVM model ===
model = joblib.load('behavior_model.pkl')

# === 2) Mediapipe setup ===
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

hands = mp_hands.Hands()
face = mp_face.FaceMesh()
pose = mp_pose.Pose()

# === 3) Webcam ===
cap = cv2.VideoCapture(0)

prev_hand = None
hand_speed = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hands
    hand_result = hands.process(rgb)
    if hand_result.multi_hand_landmarks:
        lm = hand_result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
        hand_pos = np.array([lm.x * w, lm.y * h])
        if prev_hand is not None:
            dist = np.linalg.norm(hand_pos - prev_hand)
            hand_speed = dist
        prev_hand = hand_pos
        cv2.putText(frame, f'Hand Speed: {hand_speed:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Face (eye contact & head tilt)
    face_result = face.process(rgb)
    eye_contact = 0
    head_tilt = 0
    if face_result.multi_face_landmarks:
        face_lm = face_result.multi_face_landmarks[0].landmark
        left_eye = np.array([face_lm[33].x * w, face_lm[33].y * h])
        right_eye = np.array([face_lm[263].x * w, face_lm[263].y * h])
        eye_distance = np.linalg.norm(left_eye - right_eye)
        eye_contact = 1 if eye_distance > w * 0.1 else 0  # rough threshold

        # Head tilt: vertical difference of eyes
        head_tilt = abs(left_eye[1] - right_eye[1]) / h

        cv2.putText(frame, f'Eye Contact: {eye_contact}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Head Tilt: {head_tilt:.2f}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Pose (pose distance)
    pose_result = pose.process(rgb)
    pose_distance = 0
    if pose_result.pose_landmarks:
        nose = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        pose_distance = 1 - nose.z  # z is negative closer to camera

        cv2.putText(frame, f'Pose Distance: {pose_distance:.2f}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # === 4) Predict Behavior ===
    features = np.array([[hand_speed, eye_contact, head_tilt, pose_distance]])
    behavior = model.predict(features)[0]

    cv2.putText(frame, f'Behavior: {behavior}', (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Live Behavior Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

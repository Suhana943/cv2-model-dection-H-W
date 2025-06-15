import cv2
import mediapipe as mp
import math
import time

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None
points = []
thinking = False
still_counter = 0
MOVEMENT_THRESHOLD = 5  # pixels
THINKING_FRAMES = 30    # how many still frames = thinking

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            y = int(hand_landmarks.landmark[8].y * frame.shape[0])

            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            if prev_x is not None and prev_y is not None:
                distance = math.hypot(x - prev_x, y - prev_y)
                if distance < MOVEMENT_THRESHOLD:
                    still_counter += 1
                else:
                    still_counter = 0  # moving again

                if still_counter > THINKING_FRAMES:
                    thinking = True
                else:
                    thinking = False

                if not thinking:
                    points.append((x, y))

            prev_x, prev_y = x, y

    # Draw the path
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2)

    # Show status
    status_text = "Thinking..." if thinking else "Drawing..."
    cv2.putText(frame, status_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 255) if thinking else (0, 255, 0), 3)

    cv2.imshow("Thinking Detection Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        points = []
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

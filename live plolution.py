import cv2
import numpy as np

# ===============================
# 1️⃣ Open webcam
# ===============================
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ===============================
# 2️⃣ Live loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # ✅ Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ✅ Define HSV range for haze/smoke
    lower_haze = np.array([0, 0, 180])
    upper_haze = np.array([180, 50, 255])

    # ✅ Create mask
    mask = cv2.inRange(hsv, lower_haze, upper_haze)

    # ✅ Apply mask
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # ✅ Calculate pollution percentage
    total_pixels = mask.size
    haze_pixels = cv2.countNonZero(mask)
    pollution_percent = (haze_pixels / total_pixels) * 100

    # ✅ Add percentage text to frame
    text = f"Pollution: {pollution_percent:.2f}%"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)

    # ✅ Show original with text, mask, and result
    cv2.imshow("Live Webcam with Pollution %", frame)
    cv2.imshow("Haze Mask", mask)
    cv2.imshow("Detected Haze", result)

    # ✅ Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# 3️⃣ Release resources
# ===============================
cap.release()
cv2.destroyAllWindows()

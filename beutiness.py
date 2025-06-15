import cv2
import numpy as np

class FaceBeautyModel:
    def __init__(self):
        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                                  'haarcascade_frontalface_default.xml')

    def predict(self, frame):
        """
        Detect faces and calculate a fake beauty score.
        Returns:
          - faces (list of rectangles)
          - frame with drawn rectangles
          - average beauty score
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, 
                                                   minNeighbors=5, minSize=(50, 50))

        beauty_scores = []

        for (x, y, w, h) in faces:
            # Draw face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Fake beauty score:
            # Bigger face box => higher score
            # Centered face => higher score
            area = w * h
            cx = x + w / 2
            cy = y + h / 2
            img_cx = frame.shape[1] / 2
            img_cy = frame.shape[0] / 2
            center_dist = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)

            # Normalize: bigger area, closer to center = higher score
            score = area / 1000 - center_dist / 10
            beauty_scores.append(max(score, 0))  # keep >=0

        avg_score = np.mean(beauty_scores) if beauty_scores else 0

        return faces, frame, avg_score


def main():
    # ✅ 1️⃣ Create model instance
    model = FaceBeautyModel()

    # ✅ 2️⃣ Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # ✅ 3️⃣ Predict
        faces, output_frame, avg_beauty = model.predict(frame)

        # ✅ 4️⃣ Show beauty score on frame
        cv2.putText(output_frame, f"Beauty Score: {avg_beauty:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        # ✅ 5️⃣ Show frame
        cv2.imshow("Face + Beauty Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

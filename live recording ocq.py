import csv
import cv2 as cv
import cvzone as cvz
from cvzone.HandTrackingModule import HandDetector
import time
import datetime

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector
detector = HandDetector(detectionCon=0.8)

# MCQ class
class MCQ:
    def __init__(self, data):
        self.question = data[0]
        self.choices = data[1:5]
        self.answer = int(data[5])
        self.userAns = None

    def update(self, cursor, bboxs):
        selected = False
        for i, box in enumerate(bboxs):
            x1, y1, x2, y2 = box
            if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:
                self.userAns = i + 1
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), cv.FILLED)
                selected = True
        return selected

# Load CSV
mcqList = []
with open('data.csv', newline='\n') as file:
    reader = csv.reader(file)
    data = list(reader)[1:]
    for q in data:
        mcqList.append(MCQ(q))

totalQ = min(10, len(mcqList))  # Only 10 questions max
qNo = 0
confirming = False  # Flag to prevent multiple nexts

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (1280, 720))
    hands, frame = detector.findHands(frame, flipType=True)

    dt = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    frame = cv.putText(frame, dt, (900, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    if qNo < totalQ:
        mcq = mcqList[qNo]

        # Show question & choices
        frame, _ = cvz.putTextRect(frame, f'Q: {mcq.question}', [50, 100], 2, 2, offset=20, border=2)
        bboxs = []
        for i, choice in enumerate(mcq.choices):
            pos = [100, 200 + i * 100]
            frame, box = cvz.putTextRect(
                frame, f"{i+1}) {choice}", pos, 2, 2, offset=20, border=2
            )
            bboxs.append(box)

        if hands:
            lmList = hands[0]['lmList']
            cursor = lmList[8][:2]  # Index fingertip
            cursor2 = lmList[4][:2]  # Thumb tip

            # Update choice selection freely
            mcq.update(cursor, bboxs)

            # Correct: get 3 returns!
            length, info, frame = detector.findDistance(cursor, cursor2, frame)

            # Pinch to confirm & move to next question
            if length < 40:
                if not confirming:
                    qNo += 1
                    confirming = True
                    time.sleep(0.5)
            else:
                confirming = False

    else:
        # Show final result
        score = sum(1 for m in mcqList[:totalQ] if m.answer == m.userAns)
        percent = int(score / totalQ * 100)
        frame, _ = cvz.putTextRect(frame, f'Quiz Completed!', [400, 200], 2, 2, offset=20, border=3)
        frame, _ = cvz.putTextRect(frame, f'Score: {score}/{totalQ} ({percent}%)', [400, 300], 2, 2, offset=20, border=3)
        frame, _ = cvz.putTextRect(frame, f'Press Q to Exit', [400, 400], 2, 2, offset=20, border=3)

    # Progress bar
    barVal = int(950 / totalQ * qNo)
    cv.rectangle(frame, (150, 650), (150 + barVal, 680), (0, 255, 0), cv.FILLED)
    cv.rectangle(frame, (150, 650), (1100, 680), (255, 255, 255), 3)
    frame, _ = cvz.putTextRect(frame, f'{int((qNo/totalQ)*100)}%', [1150, 640], 2, 2, offset=10, border=3)

    cv.imshow("MCQ Quiz", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#live mcq recording 
"""" 
live data """





cap.release()
cv.destroyAllWindows()

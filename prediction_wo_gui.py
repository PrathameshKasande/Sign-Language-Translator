import math
import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cnn8grps_rad1_model.h5")
model = load_model(MODEL_PATH)

capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
offset = 29

def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

print("[INFO] Headless OpenCV prediction running. Press 'ESC' to exit.")

while True:
    success, frame = capture.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    hands, _ = hd.findHands(frame, draw=False, flipType=True)
    white = np.ones((400, 400, 3), np.uint8) * 255

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        h_f, w_f, _ = frame.shape
        y1, y2 = max(0, y - offset), min(h_f, y + h + offset)
        x1, x2 = max(0, x - offset), min(w_f, x + w + offset)
        image = frame[y1:y2, x1:x2]

        if image.size > 0:
            handz, _ = hd2.findHands(image, draw=False, flipType=True)
            if handz:
                pts = handz[0]['lmList']
                os_x = ((400 - w) // 2) - 15
                os_y = ((400 - h) // 2) - 15

                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (5, 6), (6, 7), (7, 8),
                    (9, 10), (10, 11), (11, 12),
                    (13, 14), (14, 15), (15, 16),
                    (17, 18), (18, 19), (19, 20),
                    (5, 9), (9, 13), (13, 17), (0, 5), (0, 17)
                ]
                for p1, p2 in connections:
                    cv2.line(white, (pts[p1][0] + os_x, pts[p1][1] + os_y),
                             (pts[p2][0] + os_x, pts[p2][1] + os_y), (0, 255, 0), 3)

                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)

                cv2.imshow("Skeleton ROI", white)

                input_tensor = white.reshape(1, 400, 400, 3)
                prob = np.array(model.predict(input_tensor, verbose=0)[0], dtype='float32')
                ch1 = int(np.argmax(prob))

                # Resolved character mapping
                if ch1 == 0:
                    ch = 'A' if pts[4][0] < pts[6][0] else 'S'
                elif ch1 == 1:
                    ch = 'B' if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] else 'D'
                elif ch1 == 2:
                    ch = 'C' if distance(pts[12], pts[4]) > 42 else 'O'
                elif ch1 == 3:
                    ch = 'G' if distance(pts[8], pts[12]) > 72 else 'H'
                elif ch1 == 4:
                    ch = 'L'
                elif ch1 == 5:
                    ch = 'P'
                elif ch1 == 6:
                    ch = 'X'
                elif ch1 == 7:
                    ch = 'Y'
                else:
                    ch = " "

                cv2.putText(frame, f"Sign: {ch}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("Detection Feed", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()
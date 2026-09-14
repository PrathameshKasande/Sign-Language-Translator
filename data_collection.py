import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Configure storage paths
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "AtoZ_raw")
LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Create directories A through Z automatically
for letter in LETTERS:
    os.makedirs(os.path.join(BASE_DIR, letter), exist_ok=True)

# Initialize webcam and MediaPipe hand detectors
capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

current_idx = 0
offset = 15
step = 0
capturing = False
collected_in_batch = 0
SAMPLES_PER_LETTER = 180

# 21 Hand joint anatomical connection sequences
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (5, 6), (6, 7), (7, 8),                   # Index
    (9, 10), (10, 11), (11, 12),              # Middle
    (13, 14), (14, 15), (15, 16),             # Ring
    (17, 18), (18, 19), (19, 20),             # Pinky
    (5, 9), (9, 13), (13, 17), (0, 5), (0, 17) # Palm structure
]

print("[INFO] Controls: 'a' to Toggle Capturing | 'n' for Next Letter | 'ESC' to Exit")

while True:
    success, frame = capture.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    hands, _ = hd.findHands(frame, draw=False, flipType=True)
    c_dir = LETTERS[current_idx]
    save_path = os.path.join(BASE_DIR, c_dir)
    existing_count = len(os.listdir(save_path))

    # Standard white canvas (400x400x3)
    skeleton = np.ones((400, 400, 3), np.uint8) * 255

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        h_frame, w_frame, _ = frame.shape

        # Bound region within image borders
        y1, y2 = max(0, y - offset), min(h_frame, y + h + offset)
        x1, x2 = max(0, x - offset), min(w_frame, x + w + offset)
        cropped_hand = frame[y1:y2, x1:x2]

        if cropped_hand.size > 0:
            handz, _ = hd2.findHands(cropped_hand, draw=False, flipType=True)
            if handz:
                pts = handz[0]['lmList']
                os_x = ((400 - w) // 2) - 15
                os_y = ((400 - h) // 2) - 15

                # Draw joint connecting lines
                for p1, p2 in CONNECTIONS:
                    cv2.line(skeleton, (pts[p1][0] + os_x, pts[p1][1] + os_y),
                             (pts[p2][0] + os_x, pts[p2][1] + os_y), (0, 255, 0), 3)

                # Draw joint landmark coordinates
                for i in range(21):
                    cv2.circle(skeleton, (pts[i][0] + os_x, pts[i][1] + os_y), 3, (0, 0, 255), -1)

                cv2.imshow("Normalized Skeleton Canvas", skeleton)

                # Record sample on alternate steps to prevent duplicate posture frames
                if capturing and collected_in_batch < SAMPLES_PER_LETTER:
                    if step % 2 == 0:
                        file_name = f"{existing_count + 1}.jpg"
                        cv2.imwrite(os.path.join(save_path, file_name), skeleton)
                        collected_in_batch += 1
                    step += 1

                if collected_in_batch >= SAMPLES_PER_LETTER:
                    capturing = False
                    print(f"[SUCCESS] Reached 180 samples for letter: {c_dir}")

    # UI Status Overlay
    status_text = f"Class: {c_dir} | Total Saved: {existing_count}/180 | Capturing: {capturing}"
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Webcam Stream", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('a'):  # Toggle collection on/off
        capturing = not capturing
        collected_in_batch = 0
    elif key == ord('n'):  # Move to the next letter class
        current_idx = (current_idx + 1) % len(LETTERS)
        capturing = False
        collected_in_batch = 0

capture.release()
cv2.destroyAllWindows()
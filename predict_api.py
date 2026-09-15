import base64
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

# Load trained deep learning model
model = tf.keras.models.load_model("models/sign_language_model.keras")
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Alphabet classes A-Z
LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# State Buffers
current_character = ""
sentence = ""
temp_char = ""
char_hold_count = 0
CONFIRMATION_FRAMES = 6   # Hold sign for 6 consecutive frames to confirm letter
no_hand_counter = 0
SPACE_THRESHOLD = 18      # ~1.2s without hand adds a space

# Skeleton joint connections for 21 MediaPipe landmarks
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                   # Palm Base
]

def process_frame(frame):
    global current_character, sentence, temp_char, char_hold_count, no_hand_counter

    # Pure white canvas for the 2nd skeleton window
    white_skeleton = np.ones((300, 300, 3), dtype=np.uint8) * 255
    hands, img = detector.findHands(frame, draw=False)

    if hands:
        no_hand_counter = 0
        hand = hands[0]
        lmList = hand["lmList"]
        x, y, w, h = hand["bbox"]

        # 1. Normalize and draw green hand skeleton onto white canvas
        if lmList:
            all_x = [pt[0] for pt in lmList]
            all_y = [pt[1] for pt in lmList]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            span_w = max(1, max_x - min_x)
            span_h = max(1, max_y - min_y)
            scale = min(220 / span_w, 220 / span_h)

            norm_pts = []
            for pt in lmList:
                px = int(150 + (pt[0] - (min_x + span_w / 2)) * scale)
                py = int(150 + (pt[1] - (min_y + span_h / 2)) * scale)
                norm_pts.append((px, py))

            for p1, p2 in SKELETON_CONNECTIONS:
                cv2.line(white_skeleton, norm_pts[p1], norm_pts[p2], (0, 200, 0), 3)

        # 2. Crop hand region and predict character
        img_h, img_w, _ = img.shape
        y1, y2 = max(0, y - 20), min(img_h, y + h + 20)
        x1, x2 = max(0, x - 20), min(img_w, x + w + 20)
        img_crop = img[y1:y2, x1:x2]

        if img_crop.size != 0:
            img_resize = cv2.resize(img_crop, (64, 64))
            img_array = np.expand_dims(img_resize / 255.0, axis=0)

            predictions = model.predict(img_array, verbose=0)
            pred_idx = int(np.argmax(predictions))
            conf = float(predictions[0][pred_idx])

            if conf > 0.75:
                pred_char = LABELS[pred_idx]
                current_character = pred_char

                # Temporal stability: only append if held steady
                if pred_char == temp_char:
                    char_hold_count += 1
                    if char_hold_count == CONFIRMATION_FRAMES:
                        sentence += pred_char
                else:
                    temp_char = pred_char
                    char_hold_count = 0
            else:
                char_hold_count = 0
    else:
        # Hand out of frame: auto-space trigger
        current_character = ""
        char_hold_count = 0
        temp_char = ""
        no_hand_counter += 1

        if no_hand_counter == SPACE_THRESHOLD:
            if sentence and not sentence.endswith(" "):
                sentence += " "

    # Encode white skeleton to Base64 image
    _, buffer = cv2.imencode(".jpg", white_skeleton)
    skeleton_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    return {
        "character": current_character,
        "sentence": sentence,
        "skeleton": skeleton_base64
    }

def clear_all():
    global sentence, current_character, temp_char, char_hold_count
    sentence = ""
    current_character = ""
    temp_char = ""
    char_hold_count = 0

def add_space():
    global sentence
    if sentence and not sentence.endswith(" "):
        sentence += " "
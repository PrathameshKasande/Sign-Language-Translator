import os
import cv2
import numpy as np
import base64
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import enchant

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_model_26class.h5")

# Initialize models and detectors
model = load_model(MODEL_PATH, compile=False)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
dict_checker = enchant.Dict("en_US")
OFFSET = 29

def predict_frame(input_data, current_word=""):
    try:
        # Decode frame
        if isinstance(input_data, (bytes, bytearray)):
            np_arr = np.frombuffer(input_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            frame = input_data

        if frame is None:
            return {"character": "No Hand Detected", "confidence": 0.0, "skeleton": None, "suggestions": []}

        predicted_char = "No Hand Detected"
        confidence = 0.0
        white_canvas = np.ones((400, 400, 3), np.uint8) * 255

        hands, _ = hd.findHands(frame, draw=False, flipType=True)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            h_f, w_f, _ = frame.shape

            y1, y2 = max(0, y - OFFSET), min(h_f, y + h + OFFSET)
            x1, x2 = max(0, x - OFFSET), min(w_f, x + w + OFFSET)
            cropped = frame[y1:y2, x1:x2]

            if cropped.size > 0:
                handz, _ = hd2.findHands(cropped, draw=False, flipType=True)
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
                        cv2.line(white_canvas, (pts[p1][0] + os_x, pts[p1][1] + os_y),
                                 (pts[p2][0] + os_x, pts[p2][1] + os_y), (0, 255, 0), 3)

                    for i in range(21):
                        cv2.circle(white_canvas, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)

                    resized_input = cv2.resize(white_canvas, (128, 128))
                    tensor = resized_input.reshape(1, 128, 128, 3)

                    preds = model.predict(tensor, verbose=0)[0]
                    char_idx = int(np.argmax(preds))
                    confidence = float(np.max(preds))
                    predicted_char = chr(ord('A') + char_idx)

        # Base64 encode skeleton
        _, buffer = cv2.imencode('.jpg', white_canvas)
        skeleton_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        # Generate spelling suggestions based on current word
        suggestions = []
        clean_word = (current_word or "").strip()
        if clean_word:
            query = clean_word + (predicted_char if predicted_char != "No Hand Detected" else "")
            if query:
                suggestions = dict_checker.suggest(query)

        return {
            "character": predicted_char,
            "confidence": round(confidence * 100, 2),
            "skeleton": skeleton_b64,
            "suggestions": suggestions[:4]
        }

    except Exception as e:
        return {"error": str(e), "character": "No Hand Detected", "confidence": 0.0, "skeleton": None, "suggestions": []}
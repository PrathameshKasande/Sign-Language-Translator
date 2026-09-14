import math
import os
import threading
import tkinter as tk
import cv2
import enchant
import numpy as np
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
from PIL import Image, ImageTk

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sign_model_26class.h5")

ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 29

class Application:
    def __init__(self):
        self.vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.vs.isOpened():
            self.vs = cv2.VideoCapture(0)

        print(f"[INFO] Loading 26-class model from: {MODEL_PATH}")
        self.model = load_model(MODEL_PATH, compile=False)
        print("[INFO] Model loaded successfully.")

        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 120)

        self.root = tk.Tk()
        self.root.title("Sign Language To Text & Speech Conversion")
        self.root.geometry("1300x750")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root, bg="#222")
        self.panel.place(x=60, y=30, width=480, height=400)

        self.panel2 = tk.Label(self.root, bg="#fff")
        self.panel2.place(x=600, y=30, width=400, height=400)

        self.lbl_char = tk.Label(self.root, text="Character :", font=("Courier", 26, "bold"))
        self.lbl_char.place(x=60, y=470)

        self.lbl_sentence = tk.Label(self.root, text="Sentence :", font=("Courier", 26, "bold"))
        self.lbl_sentence.place(x=60, y=520)

        self.lbl_sugg = tk.Label(self.root, text="Suggestions :", font=("Courier", 20, "bold"), fg="red")
        self.lbl_sugg.place(x=60, y=580)

        self.b1 = tk.Button(self.root, text="", font=("Courier", 16), command=self.action1)
        self.b1.place(x=320, y=580)
        self.b2 = tk.Button(self.root, text="", font=("Courier", 16), command=self.action2)
        self.b2.place(x=480, y=580)
        self.b3 = tk.Button(self.root, text="", font=("Courier", 16), command=self.action3)
        self.b3.place(x=640, y=580)
        self.b4 = tk.Button(self.root, text="", font=("Courier", 16), command=self.action4)
        self.b4.place(x=800, y=580)

        self.btn_speak = tk.Button(self.root, text="Speak", font=("Courier", 18), bg="green", fg="white", command=self.speak_fun)
        self.btn_speak.place(x=1050, y=470, width=120)

        self.btn_clear = tk.Button(self.root, text="Clear", font=("Courier", 18), bg="red", fg="white", command=self.clear_fun)
        self.btn_clear.place(x=1050, y=530, width=120)

        self.str = ""
        self.word = ""
        self.word1, self.word2, self.word3, self.word4 = "", "", "", ""
        self.current_symbol = ""
        self.prev_char = ""
        self.count = 0
        self.ten_prev_char = [" "] * 10

        self.video_loop()

    def video_loop(self):
        try:
            success, frame = self.vs.read()
            if success and frame is not None:
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

                            self.predict(white, pts)

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.imgtk1 = ImageTk.PhotoImage(image=Image.fromarray(img_rgb).resize((480, 400)))
                self.panel.config(image=self.imgtk1)

                skel_rgb = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)
                self.imgtk2 = ImageTk.PhotoImage(image=Image.fromarray(skel_rgb))
                self.panel2.config(image=self.imgtk2)

        except Exception as e:
            print(f"[ERROR in video_loop] {e}")
        finally:
            self.root.after(30, self.video_loop)

    def predict(self, white_canvas, pts):
        # Resize to 128x128 to match new CNN input tensor
        canvas_resized = cv2.resize(white_canvas, (128, 128))
        input_tensor = canvas_resized.reshape(1, 128, 128, 3)

        prob = np.array(self.model.predict(input_tensor, verbose=0)[0], dtype='float32')
        char_idx = int(np.argmax(prob))
        ch = chr(ord('A') + char_idx)

        # Geometric control gestures
        if pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]:
            ch = 'Space'
        if pts[4][0] < pts[5][0] and pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
            ch = 'Next'
        if pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0] and pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1]:
            ch = 'Backspace'

        self.current_symbol = ch
        self.lbl_char.config(text=f"Character : {self.current_symbol}")

        if self.current_symbol == "Next" and self.prev_char != "Next":
            candidate = self.ten_prev_char[(self.count - 2) % 10]
            if candidate == "Backspace":
                self.str = self.str[:-1]
            elif candidate == "Space":
                self.str += " "
            elif candidate not in ["Next", "Backspace", "Space", " "]:
                self.str += candidate

            self.lbl_sentence.config(text=f"Sentence : {self.str}")
            self.update_suggestions()

        self.prev_char = self.current_symbol
        self.count += 1
        self.ten_prev_char[self.count % 10] = self.current_symbol

    def update_suggestions(self):
        last_space = self.str.rfind(" ")
        self.word = self.str[last_space + 1:].strip()
        if self.word:
            suggestions = ddd.suggest(self.word)
            self.word1 = suggestions[0] if len(suggestions) > 0 else ""
            self.word2 = suggestions[1] if len(suggestions) > 1 else ""
            self.word3 = suggestions[2] if len(suggestions) > 2 else ""
            self.word4 = suggestions[3] if len(suggestions) > 3 else ""
        else:
            self.word1, self.word2, self.word3, self.word4 = "", "", "", ""

        self.b1.config(text=self.word1)
        self.b2.config(text=self.word2)
        self.b3.config(text=self.word3)
        self.b4.config(text=self.word4)

    def apply_word(self, selected_word):
        if not selected_word:
            return
        last_space = self.str.rfind(" ")
        self.str = self.str[:last_space + 1] + selected_word.upper() + " "
        self.lbl_sentence.config(text=f"Sentence : {self.str}")
        self.update_suggestions()

    def action1(self): self.apply_word(self.word1)
    def action2(self): self.apply_word(self.word2)
    def action3(self): self.apply_word(self.word3)
    def action4(self): self.apply_word(self.word4)

    def speak_fun(self):
        text = self.str.strip()
        if text:
            threading.Thread(target=lambda: (self.speak_engine.say(text), self.speak_engine.runAndWait()), daemon=True).start()

    def clear_fun(self):
        self.str = ""
        self.word1, self.word2, self.word3, self.word4 = "", "", "", ""
        self.lbl_sentence.config(text="Sentence : ")
        self.b1.config(text="")
        self.b2.config(text="")
        self.b3.config(text="")
        self.b4.config(text="")

    def destructor(self):
        self.vs.release()
        self.root.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = Application()
    app.root.mainloop()
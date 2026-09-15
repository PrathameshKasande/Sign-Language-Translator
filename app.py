import base64
import os
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
import predict_api

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/translate")
def translate():
    return render_template("translate.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image frame received"}), 400

    try:
        header, encoded = data["image"].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        result = predict_api.process_frame(frame)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear():
    predict_api.clear_all()
    return jsonify({"status": "cleared"})

@app.route("/space", methods=["POST"])
def space():
    predict_api.add_space()
    return jsonify({"status": "space_added"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
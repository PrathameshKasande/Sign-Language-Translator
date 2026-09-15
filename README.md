# 🤟 Sign Language to Text & Speech Translator

![Python Version](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.0-red?logo=keras)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?logo=flask)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![Accuracy](https://img.shields.io/badge/Model_Accuracy-92%25-brightgreen)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions)

An end-to-end computer vision and deep learning web platform engineered to bridge communication barriers for the **Deaf and Hard of Hearing (DHH) community**. The application captures real-time hand gestures via any standard webcam, translates sign language into text, and synthesizes voice feedback with high reliability.

Live Link : https://sign-language-translator-sa97.onrender.com

---

## 🎯 Problem Statement

Everyday verbal communication poses a significant barrier for the deaf and mute community, as the general public rarely understands formal sign language. Traditional communication methods (such as physical note-writing or hiring certified interpreters) are slow, inaccessible during emergencies, and often cost-prohibitive.

This project delivers an automated, contactless, real-time sign language interpreter accessible directly through any web browser—democratizing two-way dialogue without requiring specialized sensors or dedicated hardware.

---

## ✨ Key Features

* **Real-Time Hand Landmark Detection:** Uses MediaPipe and OpenCV to isolate and track 21 distinct hand landmarks per frame with low latency.
* **Deep Learning Classifier (92% Accuracy):** Built with TensorFlow and Keras 3.0, trained to recognize multi-class alphabetic hand configurations with high precision.
* **Browser-Based Webcam Stream:** Captures frames client-side using JavaScript and sends lightweight asynchronous payloads to Flask REST endpoints for real-time inference.
* **Integrated Text-to-Speech (TTS):** Converts predicted sign gestures into spoken audio output (`espeak` / speech synthesis) for seamless two-way interactions.
* **Production-Ready Docker Stack:** Encapsulated in a lightweight `python:3.11-slim` container running production-grade Gunicorn WSGI.
* **Automated CI/CD:** GitHub Actions test pipeline automatically verifies container integrity, dependency isolation, and endpoint smoke tests on every push.

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Deep Learning & CV** | TensorFlow 2.16+, Keras 3.0, MediaPipe, OpenCV (`opencv-python-headless`) |
| **Backend & API** | Python 3.11, Flask, Gunicorn |
| **Frontend** | HTML5, CSS3, JavaScript (WebRTC / Canvas Stream API) |
| **Audio Synthesis** | eSpeak, FFmpeg |
| **DevOps & Cloud** | Docker, GitHub Actions (CI/CD Pipeline), Render PaaS |

---

## 📊 Model Performance

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | **92%** |
| **Inference Latency** | ~35ms – 50ms per frame (CPU-optimized) |
| **Landmark Points Tracked** | 21 Hand Knuckle Coordinates |
| **Classification Domain** | American Sign Language (ASL) Static Gestures |

---

## 📂 Project Structure

```text
Sign-Language-Translator/
│
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI/CD Docker build & smoke-test pipeline
│
├── data/
│   └── AtoZ_raw/                 # Raw image dataset organized by letter classes (A-Z)
│
├── models/
│   └── sign_language_model.keras # Trained Keras/TensorFlow deep learning model weights
│
├── templates/
│   ├── index.html                # Main webcam translation UI & dashboard
│   ├── login.html                # User authentication page
│   └── register.html             # User registration page
│
├── app.py                        # Core Flask application, auth routing, & video stream endpoints
├── predict_api.py                # Real-time inference engine, MediaPipe landmark extraction logic
├── prediction_wo_gui.py          # Standalone CLI inference script for local terminal testing
├── final_pred.py                 # Final prediction and sentence formulation module
│
├── data_collection.py            # Script for capturing custom gesture dataset via OpenCV
├── model_trainer.py              # Model architecture definition & training pipeline
├── evaluate_model.py             # Evaluation script to compute test loss, accuracy & confusion matrix
│
├── hand_landmarker.task          # Pre-packaged MediaPipe landmark task asset
├── users.json                    # Local user credentials store
├── requirements.txt              # Production Python package dependencies
├── Dockerfile                    # Container configuration with Debian system dependencies
├── .dockerignore                 # Excludes caches, venv, and datasets from Docker image
├── .gitignore                    # Excludes virtual environments and temporary files from Git
└── .gitattributes                # Repository line-ending and Git LFS configurations



💻 How to Run the Project (Step-by-Step)
Prerequisites
Ensure you have the following installed on your machine:

Git: Download Git

Python 3.11 (recommended): Download Python

Docker Desktop (optional, for running via container): Download Docker

A functional webcam connected to your system.

Method 1: Local Setup (Recommended for Development)
Clone the repository:

Bash
git clone [https://github.com/...../Sign-Language-Translator.git
cd Sign-Language-Translator
Set up a Python virtual environment:

On Windows (Command Prompt / PowerShell):

python -m venv venv
venv\Scripts\activate


### Install Python dependencies:
---

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

---

Start the Flask server:

python app.py
Access the web app:

Open your browser and navigate to: http://localhost:5000 (or http://127.0.0.1:5000).



Method 2: Run via Docker (Zero Configuration)
If you have Docker Desktop installed, you can build and run the entire environment inside an isolated container without installing Python libraries manually:

Clone and enter the directory:


Build the Docker image:

docker build -t sign-language-translator .

Run the container:

docker run -d -p 5000:5000 --name sl_translator sign-language-translator

Open in browser:
Navigate to http://localhost:5000 in your web browser.

Stop the container when finished:

docker stop sl_translator

Method 3: Run Without GUI (Terminal CLI Mode)
If you want to run quick predictions directly in your terminal using OpenCV without running the Flask web server:
---
python prediction_wo_gui.py
---
Press q in the OpenCV camera window to exit.



🚀 Roadmap & Future Enhancements
Dynamic Sign Recognition (LSTM / GRU): Transition from static alphabet classification to temporal sequence modeling for whole words and common phrases ("Hello", "Thank you").

Two-Hand Landmark Extraction: Expand the MediaPipe input pipeline to detect and track bilateral gestures simultaneously.

NLP Auto-Correction Layer: Integrate language modeling (e.g., n-gram or pyspellchecker) to assemble detected characters into grammatically correct sentences.

Client-Side Edge Inference (ONNX / TF.js): Compile model weights to run directly in the browser, eliminating server roundtrips and keeping inference latency under 20ms.

Secure Database Layer: Migrate local credential tracking from users.json to MongoDB / PostgreSQL with salted bcrypt password hashing.

🤝 Accessibility & Community Impact
This application was engineered with a primary focus on human-centered design for the Deaf and Hard of Hearing community:

Enables direct communication without requiring third-party human interpreters.

Operates on standard commodity webcams and budget consumer hardware.

Designed to run completely on CPU infrastructure to keep hosting and access free and scalable.

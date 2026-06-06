# Real-Time Sign Language Recognition and Translation System

**Developed by:** Mohamed (moh301)

An end-to-end Assistive Technology platform designed to bridge the communication gap for the deaf and mute community. This system captures sign language gestures via a standard webcam, extracts biological hand features in real-time, and translates them into text and multi-lingual speech.

---

## Key Features

* **Real-Time Landmark Extraction:** Powered by Google MediaPipe to extract 21 points (63 coordinates) from the hand, rendering the system invariant to background noise or lighting conditions.
* **Intelligent Handling of Empty States:** The system explicitly learns a NO_HAND state to completely eliminate false predictions when no user is in the frame.
* **Prediction Smoothing Buffer:** Implements a temporal queue (deque mechanism) that filters high-frequency jitters, ensuring stable and smooth on-screen text transitions.
* **Creator Mode:** Allows users to record sign videos, auto-save metadata, and maintain a custom translation dictionary.
* **Multi-Lingual Text-To-Speech (TTS):** Instant audio translation supporting English, Arabic, French, and Spanish.

---

## AI Model Architecture

The core AI engine is a Deep Multi-Layer Perceptron (MLP) built with TensorFlow and Keras:
* **Data Augmentation:** Gaussian Noise injection to ensure model robustness against hand tremors.
* **Overfitting Prevention:** Stacked BatchNormalization and Dropout layers combined with EarlyStopping callbacks to guarantee generalizability.

---

## Tech Stack

* **Backend:** Flask (Python)
* **Computer Vision & AI:** MediaPipe, TensorFlow, OpenCV, Scikit-Learn
* **Audio:** gTTS (Google Text-To-Speech)

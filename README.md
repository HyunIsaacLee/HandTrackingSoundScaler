# Hand Tracking Sound Scaler

> Control music playback (volume, pitch, speed) using hand gestures in real time.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)

---

## Project Overview
This project uses **MediaPipe** for hand tracking and **SoundDevice + Librosa** for real-time audio control.

- **Left hand** controls **pitch**  
- **Right hand** controls **speed**  
- **Distance between hands** controls **volume**

The program visualizes hand landmarks using OpenCV while modifying the audio in real time.

---

## Features
- Real-time hand tracking with OpenCV + MediaPipe
- Volume, pitch, and speed control of `.wav` audio files
- .wav file: https://freesound.org/people/bassimat/sounds/824892/
- Visualization of hand landmarks for easy tracking

---

## Installation

```bash
# Clone the repo
git clone https://https://github.com/HyunIsaacLee/HandTrackingSoundScaler.git
cd HandTrackingSoundScaler

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

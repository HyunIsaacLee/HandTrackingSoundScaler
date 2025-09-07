import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import math
import threading
import librosa
import soundfile as sf
import sounddevice as sd

######################
wCam, hCam = 640, 480
######################



pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detectionCon=0.7)
samples, samplerate = sf.read("824892__bassimat__piano-loop-fugue-machine-120bpm-001.wav", dtype='float32')
idx = 0
volume = 0.50
speed = 0.5
pitch = 0.5
minVol = 0.00
maxVol = 2.00
minPitch = -2.00
maxPitch = 2.00
minSpeed = 0.01
maxSpeed = 2.00

def apply_volume(chunk, volume):
    return chunk * volume

def apply_speed(chunk, speed, samplerate):
    return librosa.resample(
        chunk.T,
        orig_sr=int(samplerate * speed),
        target_sr=samplerate
    ).T

def apply_pitch(chunk, pitch, samplerate):
    # librosa works on mono, so handle both mono & stereo
    if len(chunk.shape) > 1:  # stereo
        shifted = []
        for ch in range(chunk.shape[1]):
            shifted_ch = librosa.effects.pitch_shift(chunk[:, ch], sr=samplerate, n_steps=pitch)
            shifted.append(shifted_ch)
        return np.stack(shifted, axis=1)
    else:  # mono
        return librosa.effects.pitch_shift(chunk, sr=samplerate, n_steps=pitch)

def callback(outdata, frames, time_info, status):
    global idx, speed, volume, pitch
    if status:
        print(status)

    step = int(frames * speed)
    if step <= 0:
        step = frames  # safeguard against zero or negative

    chunk = samples[idx:idx+step]
    idx += step

    if len(chunk) < step:
        idx = 0
        if len(samples.shape) > 1:
            chunk = np.pad(chunk, ((0, step - len(chunk)), (0, 0)), mode="constant")
        else:
            chunk = np.pad(chunk, (0, step - len(chunk)), mode="constant")

    # Apply effects in order
    chunk = apply_speed(chunk, speed, samplerate)
    chunk = apply_pitch(chunk, pitch, samplerate)
    chunk = apply_volume(chunk, volume)

    # Ensure correct shape for output
    if len(samples.shape) > 1:
        outdata[:len(chunk), :] = chunk
    else:
        outdata[:len(chunk), 0] = chunk


stream = sd.OutputStream(
    samplerate=samplerate,
    channels=samples.shape[1] if len(samples.shape) > 1 else 1,
    callback=callback
)

stream.start()

#LOOPS EVERY FRAME
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    count = 0
    if detector.results.multi_hand_landmarks:
        for handNo in range(len(detector.results.multi_hand_landmarks)):
            lmList = detector.findPosition(img, handNo=handNo, draw=False)
            count = count + 1
            if len(lmList) != 0:
                x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
                x2, y2 = lmList[8][1], lmList[8][2]  # Index tip
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(img, (cx, cy), 3, (255, 255, 255), cv2.FILLED)
                cv2.circle(img, (x1, y1), 3, (255, 255, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 3, (255, 255, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (128, 128, 128), 1)

                Length = math.hypot(x2 - x1, y2 - y1)
                print(f"this is a the smaller length number {count}: {Length}")

                if Length < 21:
                    cv2.circle(img, (cx, cy), 3, (64, 64, 64), cv2.FILLED)
                if Length > 140:
                    cv2.circle(img, (cx, cy), 3, (64, 64, 64), cv2.FILLED)

                #GETS THE HAND NUMBERS
                if handNo == 0:
                    lmx, lmy = cx, cy
                    pitch = np.interp(Length, [21, 140], [minPitch, maxPitch])
                    cv2.putText(img, f"Pitch: {pitch:.2f}", (lmx-30, lmy - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                if handNo == 1:
                    rmx, rmy = cx, cy
                    speed = np.interp(Length, [21, 140], [minSpeed, maxSpeed])
                    cv2.putText(img, f"Speed: {speed:.2f}", (rmx-30, rmy - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        #IF THERE ARE TWO HANDS ON THE BOARD
        if count == 2:
            mcx, mcy = (lmx + rmx) // 2, (lmy + rmy) // 2
            cv2.circle(img, (mcx, mcy), 3, (64, 64, 64), cv2.FILLED)

            cv2.line(img, (lmx, lmy), (rmx, rmy), (128, 128, 128), 1)
            middleLength = math.hypot(rmx - lmx, rmy - lmy)
            print(f"this is a middle length Number: {middleLength}")

            volume = np.interp(middleLength, [20, 400], [minVol, maxVol])
            print(f"this is a volume number: {volume}")
            cv2.putText(img, f"Volume: {volume:.2f}", (mcx-30, mcy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if middleLength < 20:
                cv2.circle(img, (mcx, mcy), 2, (100, 255, 0), cv2.FILLED)
            if middleLength > 400:
                cv2.circle(img, (mcx, mcy), 2, (255, 255, 0), cv2.FILLED)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2 ,(255, 0, 0), 1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

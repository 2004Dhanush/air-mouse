import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Default webcam
screen_w, screen_h = pyautogui.size()
clicking = False

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)
            
            cv2.circle(frame, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (tx, ty), 15, (0, 0, 255), cv2.FILLED)
            
            sx = int(np.interp(ix, [100, w-100], [0, screen_w]))
            sy = int(np.interp(iy, [100, h-100], [0, screen_h]))
            pyautogui.moveTo(sx, sy)
            print(f"Move to: ({sx}, {sy})")  # Debug print
            
            dist = math.hypot(ix - tx, iy - ty)
            if dist < 50:
                if not clicking:
                    pyautogui.mouseDown()
                    clicking = True
                cv2.putText(frame, 'CLICK/DRAG', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                if clicking:
                    pyautogui.mouseUp()
                    clicking = False
    
    cv2.imshow('Air Mouse - Complete', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

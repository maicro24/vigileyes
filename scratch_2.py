import cv2
import numpy as np
import pygame
import os
import mediapipe as mp
import time
from collections import deque

def eye_aspect_ratio(landmarks, eye_indices):
    vertical1 = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
    vertical2 = (np.linalg.norm
                 (np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]])))
    horizontal = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
    return (vertical1 + vertical2) / (2.0 * horizontal)

def draw_eye_contour(frame, landmarks, eye_indices, color=(0, 255, 0), thickness=2):
    points = np.array([landmarks[i] for i in eye_indices], dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)

def draw_eye_rectangle(frame, landmarks, eye_indices, color=(0, 255, 0), thickness=2):
    points = np.array([landmarks[i] for i in eye_indices], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(points)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

def main():
    pygame.mixer.init()

    SOUND_FILE = "alarm.wav"
    if not os.path.exists(SOUND_FILE):
        print("Erreur : le fichier alarm.wav est introuvable.")
        return

    alarm_sound = pygame.mixer.Sound(SOUND_FILE)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    CLOSED_EYES_CONSEC_FRAMES = 10
    EAR_THRESHOLD = 0.25
    COUNTER = 0
    ALARM_ON = False
    avg_ear = 0.0

    blink_counter = 0
    blink_timestamps = deque()
    DROWSY_BLINK_RATE = 20
    TIME_WINDOW = 60

    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                             for lm in results.multi_face_landmarks[0].landmark]

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                current_ear = (left_ear + right_ear) / 2.0
                avg_ear = 0.8 * avg_ear + 0.2 * current_ear

                draw_eye_contour(frame, landmarks, LEFT_EYE)
                draw_eye_contour(frame, landmarks, RIGHT_EYE)
                draw_eye_rectangle(frame, landmarks, LEFT_EYE)
                draw_eye_rectangle(frame, landmarks, RIGHT_EYE)

                if avg_ear < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= CLOSED_EYES_CONSEC_FRAMES and not ALARM_ON:
                        alarm_sound.play(-1)
                        ALARM_ON = True
                else:
                    if 0 < COUNTER < CLOSED_EYES_CONSEC_FRAMES:
                        blink_counter += 1
                        blink_timestamps.append(time.time())
                        now = time.time()
                        while blink_timestamps and now - blink_timestamps[0] > TIME_WINDOW:
                            blink_timestamps.popleft()
                        # 👇 إشعار بأن الرمش طبيعي
                        cv2.putText(frame, "Blink OK", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    COUNTER = 0
                    if ALARM_ON:
                        alarm_sound.stop()
                        ALARM_ON = False

                # معلومات العرض
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Blinks: {blink_counter}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                current_blink_rate = len(blink_timestamps) / (TIME_WINDOW / 60) if TIME_WINDOW > 0 else 0
                cv2.putText(frame, f"Blink Rate: {current_blink_rate:.1f}/min", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if current_blink_rate > DROWSY_BLINK_RATE:
                    cv2.putText(frame, "DROWSY: High Blink Rate!", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Driver Sleep Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ALARM_ON:
            alarm_sound.stop()
        pygame.quit()

if __name__ == "__main__":
    main()

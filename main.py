import cv2
import mediapipe as mp
from typing import Final
import pyautogui

RETURN_KEY: Final = 27  # this is for esc key
MASK: Final = 0xFF
ACTION_FINGERS = 2  # at this point I only want to move through the screen


def is_finger_down(landmarks, finger_tips, finger_mcp):
    return landmarks[finger_tips].y > landmarks[finger_mcp].y


def main():
    mp_hands = mp.solutions.hands
    mp_lines = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)  # default camera

    with mp_hands.Hands(
        min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=1
    ) as model:
        fingers_state = [False] * 6

        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.process(rgb_frame)

            if results.multi_hand_landmarks:
                for h, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_lines.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    finger_tips = [12, 16]
                    finger_mcp = [9, 13]
                    for i in range(ACTION_FINGERS):
                        finger_index = i + h * ACTION_FINGERS
                        down= is_finger_down(
                            hand_landmarks.landmark, finger_tips[i], finger_mcp[i]
                        )
                        if down :
                            if not fingers_state[finger_index]:
                                print("bajé")
                                pyautogui.write("hello")
                                fingers_state[finger_index] = True
                        else:
                            fingers_state[finger_index] =False
                            print("subí")

            cv2.imshow("Hands", frame)

            if cv2.waitKey(1) & MASK == RETURN_KEY:
                break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

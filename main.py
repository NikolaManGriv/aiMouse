import cv2
import mediapipe as mp
from typing import Final
import pyautogui

RETURN_KEY: Final = 27  # this is for esc key
MASK: Final = 0xFF
INDEX_TIP: Final = 8
INDEX_MCP: Final = 5
PINKY_TIP: Final = 20
PINKY_MCP: Final= 17

def is_finger_down(landmarks, finger_tips, finger_mcp):
    return landmarks[finger_tips].y > landmarks[finger_mcp].y


def main():
    mp_hands = mp.solutions.hands
    mp_lines = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)  # default camera

    with mp_hands.Hands(
        min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=1
    ) as model:

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
                
                    index_down= is_finger_down(hand_landmarks.landmark, INDEX_TIP, INDEX_MCP)
                    pinky_down= is_finger_down(hand_landmarks.landmark, PINKY_TIP, PINKY_MCP) 

                    if not index_down and pinky_down:
                        pyautogui.press('down')
                    elif not index_down and not pinky_down :
                        pyautogui.press('up')

            cv2.imshow("Hands", frame)

            if cv2.waitKey(1) & MASK == RETURN_KEY:
                break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

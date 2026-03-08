import cv2
import mediapipe as mp
from typing import Final

RETURN_KEY: Final = 27  # this is for esc key
MASK: Final = 0xFF


def main():
    mp_hands = mp.solutions.hands
    mp_lines = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)  # default camera

    with mp_hands.Hands(
        min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=1
    ) as model:
        # fingers_state = [False] * 6

        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_lines.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            cv2.imshow("Hands", frame)

            if cv2.waitKey(1) & MASK == RETURN_KEY:
                break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

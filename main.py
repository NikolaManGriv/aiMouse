import cv2
import mediapipe as mp
from typing import Final
import pyautogui

RETURN_KEY: Final = 27  # this is for esc key
MASK: Final = 0xFF
INDEX_TIP: Final = 8
INDEX_MCP: Final = 5
PINKY_TIP: Final = 20
PINKY_MCP: Final = 17
MIDDLE_TIP: Final = 12
MIDDLE_MCP: Final = 9
RING_TIP: Final = 16
RING_MCP: Final = 13
SMOOTHING_FACTOR = 5  # 20%


def is_finger_down(landmarks, finger_tips, finger_mcp):
    return landmarks[finger_tips].y > landmarks[finger_mcp].y


def main():
    mp_hands = mp.solutions.hands
    mp_lines = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)  # default camera
    screen_width, screen_height = pyautogui.size()

    with mp_hands.Hands(
        min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=1
    ) as model:
        prev_x, prev_y = 0, 0
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

                    index_down = is_finger_down(
                        hand_landmarks.landmark, INDEX_TIP, INDEX_MCP
                    )
                    pinky_down = is_finger_down(
                        hand_landmarks.landmark, PINKY_TIP, PINKY_MCP
                    )
                    middle_down = is_finger_down(
                        hand_landmarks.landmark, MIDDLE_TIP, MIDDLE_MCP
                    )
                    ring_down = is_finger_down(
                        hand_landmarks.landmark, RING_TIP, RING_MCP
                    )

                    go_down = (
                        not index_down and pinky_down and middle_down and ring_down
                    )
                    go_up = (
                        not index_down and not pinky_down and middle_down and ring_down
                    )
                    move_mouse = (
                        not index_down and pinky_down and not middle_down and ring_down
                    )
                    drag_activated = (
                        not index_down
                        and not pinky_down
                        and not middle_down
                        and ring_down
                    )
                    highlight = (
                        not index_down
                        and not middle_down
                        and not ring_down
                        and pinky_down
                    )

                    if go_down:
                        pyautogui.press("down")

                    elif go_up:
                        pyautogui.press("up")

                    elif move_mouse:
                        middle_coords_x, middle_coords_y = (
                            hand_landmarks.landmark[MIDDLE_TIP].x,
                            hand_landmarks.landmark[MIDDLE_TIP].y,
                        )
                        mouse_x, mouse_y = (
                            int(middle_coords_x * screen_width),
                            int(middle_coords_y * screen_height),
                        )
                        curr_mouse_x, curr_mouse_y = (
                            mouse_x + (mouse_x - prev_x) / SMOOTHING_FACTOR,
                            mouse_y + (mouse_y - prev_y) / SMOOTHING_FACTOR,
                        )
                        pyautogui.PAUSE = 0
                        pyautogui.moveTo(curr_mouse_x, curr_mouse_y)
                        prev_x, prev_y = curr_mouse_x, curr_mouse_y

                    elif drag_activated:
                        pyautogui.mouseDown(button="left")

                    elif highlight:
                        pyautogui.mouseUp(button="left")

            cv2.imshow("Hands", frame)

            if cv2.waitKey(1) & MASK == RETURN_KEY:
                break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

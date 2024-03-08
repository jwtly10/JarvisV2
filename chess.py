import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

frame_counter = 0
chess_overlay = None


def draw_chess_board(img, top_left_corner, size=100, squares=8, tilt_factor=0.2):
    board_size = size * squares
    board = np.zeros((board_size, board_size, 3), dtype=np.uint8)

    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                board[i * size : (i + 1) * size, j * size : (j + 1) * size] = (
                    255,
                    255,
                    255,
                )  # White square
            else:
                board[i * size : (i + 1) * size, j * size : (j + 1) * size] = (
                    0,
                    0,
                    0,
                )  # Black square

    pts1 = np.float32(
        [
            [top_left_corner[0], top_left_corner[1]],
            [top_left_corner[0] + board_size - 1, top_left_corner[1]],
            [top_left_corner[0], top_left_corner[1] + board_size - 1],
            [top_left_corner[0] + board_size - 1, top_left_corner[1] + board_size - 1],
        ]
    )

    pts2 = np.float32(
        [
            [top_left_corner[0] + board_size * tilt_factor, top_left_corner[1]],
            [top_left_corner[1] + board_size * (1 - tilt_factor), top_left_corner[1]],
            [top_left_corner[0], top_left_corner[1] + board_size - 1],
            [top_left_corner[0] + board_size - 1, top_left_corner[1] + board_size - 1],
        ]
    )

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    transformed_board = cv2.warpPerspective(board, matrix, (board_size, board_size))

    for i in range(transformed_board.shape[0]):
        for j in range(transformed_board.shape[1]):
            if np.any(transformed_board[i, j] != 0):
                img[top_left_corner[1] + i, top_left_corner[0] + j] = transformed_board[
                    i, j
                ]


while True:
    frame_counter += 1
    success, image = cap.read()
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    blank_img = np.zeros_like(image)

    results = hands.process(rgb_image)

    if chess_overlay is None or frame_counter % 10 == 0:
        chess_overlay = np.zeros_like(blank_img)
        draw_chess_board(blank_img, top_left_corner=(50, 50), size=50)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                blank_img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )

            h, w, _ = blank_img.shape
            min_x, min_y = w, h
            max_x, max_y = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                min_x, min_y = min(x, min_x), min(y, min_y)
                max_x, max_y = max(x, max_x), max(y, max_y)

            cv2.rectangle(
                blank_img,
                (min_x - 5, min_y - 5),
                (max_x + 5, max_y + 5),
                (0, 255, 0),
                2,
            )

    # This draws on the camera feed
    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         mp_draw.draw_landmarks(
    #             image,
    #             hand_landmarks,
    #             mp_hands.HAND_CONNECTIONS,
    #             mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
    #             mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
    #         )

    #         h, w, _ = image.shape
    #         min_x, min_y = w, h
    #         max_x, max_y = 0, 0
    #         for lm in hand_landmarks.landmark:
    #             x, y = int(lm.x * w), int(lm.y * h)
    #             min_x, min_y = min(x, min_x), min(y, min_y)
    #             max_x, max_y = max(x, max_x), max(y, max_y)

    #         cv2.rectangle(
    #             image, (min_x - 5, min_y - 5), (max_x + 5, max_y + 5), (0, 255, 0), 2
    #         )

    # draw_square(image, top_left_corner=(250, 70))

    # draw_perspective_square_overlay(image, top_left_corner=(250, 70))
    # draw_chess_board(image, top_left_corner=(50, 50), size=50)

    # cv2.imshow("Hand Tracking", image)

    display = blank_img.copy()

    mask = np.any(chess_overlay != [0, 0, 0], axis=-1)
    display[mask] = chess_overlay[mask]

    cv2.imshow("Hand Tracking", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
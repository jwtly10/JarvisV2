import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

circle_pos = np.array([300, 300]) 
circle_radius = 20
circle_color = (255, 0, 0)
dragging = False

bin_pos = (600, 50)
bin_color = (0, 0, 255)
in_bin = False  

while True:
    success, image  = cap.read()
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_image)


    if np.linalg.norm(circle_pos - bin_pos) < circle_radius:
        if not dragging:
            circle_color = bin_color


    if not results.multi_hand_landmarks:
        dragging = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )

            thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image.shape[1],
                                  hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image.shape[0]])
            index_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1],
                                  hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]])
            pinch_distance = np.linalg.norm(thumb_tip - index_tip)


            if pinch_distance < 60: 
                if not dragging:
                    if np.linalg.norm(thumb_tip - circle_pos) < circle_radius:
                        dragging = True
                if dragging:
                    circle_pos = (thumb_tip + index_tip) / 2
            else:
                dragging = False

            h, w, _ = image.shape
            min_x, min_y = w, h
            max_x, max_y = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                min_x, min_y = min(x, min_x), min(y, min_y)
                max_x, max_y = max(x, max_x), max(y, max_y)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinch_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

            gesture = ""
            if pinch_distance < 0.05:
                gesture = "Pinch"

            cv2.rectangle(image, (min_x - 5, min_y - 5), (max_x + 5, max_y + 5), (0, 255, 0), 2)
            if gesture:
                cv2.putText(image, gesture, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.circle(image, (int(circle_pos[0]), int(circle_pos[1])), circle_radius, circle_color, -1)

    cv2.rectangle(image, (bin_pos[0] - 50, bin_pos[1] - 50), (bin_pos[0] + 50, bin_pos[1] + 50), bin_color, 2)


    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
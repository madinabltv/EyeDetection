import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def mid_point(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

font = cv2.FONT_HERSHEY_DUPLEX

def get_blink_correlation(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    eyeline_horizontal = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)

    top_point = mid_point(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    bottom_point = mid_point(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    eyeline_vertical = cv2.line(frame, top_point, bottom_point, (0, 255, 0), 2)

    eyeline_vertical_length = hypot((top_point[0] - bottom_point[0]), (top_point[1] - bottom_point[1]))
    eyeline_horizontal_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))

    correlation = eyeline_horizontal_length/eyeline_vertical_length
    return correlation


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 4)
        landmarks = predictor(gray, face)
        # x = landmarks.part(36).x
        # y = landmarks.part(36).y
        # cv2.circle(frame, (x, y), 3, (0, 255, 0), 2)
        # left_point = (landmarks.part(36).x, landmarks.part(36).y)
        # right_point = (landmarks.part(39).x, landmarks.part(39).y)
        #
        # eyeline_horizontal = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        #
        # top_point = mid_point(landmarks.part(37), landmarks.part(38))
        # bottom_point = mid_point(landmarks.part(41), landmarks.part(40))
        #
        # eyeline_vertical = cv2.line(frame, top_point, bottom_point, (0, 255, 0), 2)
        #
        # eyeline_vertical_length = hypot((top_point[0] - bottom_point[0]), (top_point[1] - bottom_point[1]))
        # eyeline_horizontal_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        #
        # correlation = eyeline_horizontal_length/eyeline_vertical_length
        left_eye_correlation = get_blink_correlation([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_correlation = get_blink_correlation([42, 43, 44, 45, 46, 47], landmarks)

        if right_eye_correlation > 6 or left_eye_correlation > 6:
            cv2.putText(frame, "blinking", (50, 150), font, 2, (255, 0, 0))


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 2:
        break

cap.release()
cv2.destroyAllWindows()

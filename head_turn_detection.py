# importing dependencies

import cv2
import mediapipe as mp
import numpy as np
import math

# initialize mediapipe face mesh extractor

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.6,
    min_tracking_confidence = 0.7
)

# drawing utilities

mp_drawing = mp.solutions.drawing_utils
drawing_specs = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)

# function to calculate angles between 2 points

def calc_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ab = a-b
    cb = c-b
    cos_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)) # re-arrange dot product formula
    angle = np.degrees(np.arccos(cos_angle))
    return angle

# estimate head pose using key points

def estimate_head_pose(landmarks, image_w, image_h):

    # 3D points of key landmarks
    left_eye = np.array([landmarks[33].x * image_w, landmarks[33].y * image_h, landmarks[33].z])
    right_eye = np.array([landmarks[263].x * image_w, landmarks[263].y * image_h, landmarks[263].z])
    nosetip = np.array([landmarks[1].x * image_w, landmarks[1].y * image_h, landmarks[1].z])
    mouth_left = np.array([landmarks[61].x * image_w, landmarks[61].y * image_h, landmarks[61].z])
    mouth_right = np.array([landmarks[291].x * image_w, landmarks[291].y * image_h, landmarks[291].z])
    chin = np.array([landmarks[199].x * image_w, landmarks[199].y * image_h, landmarks[199].z])

    # calculate center points
    mid_eye = (left_eye + right_eye) / 2
    mid_mouth = (mouth_left + mouth_right) / 2

    # head movement estimation
    # yaw
    yaw = (nosetip[0] - mid_eye[0]) / (right_eye[0] - left_eye[0]) # relative position of nose with respect to the middle of left & right eyes

    # pitch
    pitch = (nosetip[1] - mid_mouth[1] / chin[1] - mid_eye[1]) # relative position of nose with respect to middle of eyes & mouth

    # roll 
    delta_y = right_eye[1] - left_eye[1]
    delta_x = right_eye[0] - left_eye[0]
    roll = math.degrees(math.atan2(delta_y, delta_x)) # angle the line joining both eyes makes with the x-axis

    return yaw, pitch, roll

# main driver code
def head_rotation_detector():
    turn_test_flag = 0
    left_test = 0
    right_test = 0
    up_test = 0
    down_test = 0
    cap = cv2.VideoCapture(0)
    print('Head rotation test started . . .')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('failed to capture frame !')
            break
        
        frame = cv2.flip(frame, 1) # horiziontal flipping such that movements match camera o/p (reversed by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                # draw mesh
                mp_drawing.draw_landmarks(
                    image = frame, 
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = drawing_specs
                )

                # get yaw, pitch and roll
                yaw, pitch, roll = estimate_head_pose(face_landmarks.landmark, w, h)
                # interpret directions
                direction = 'center'
                if yaw > 0.05:
                    direction = 'Looking Right'
                    left_test = 1
                elif yaw < -0.05:
                    direction = 'looking Left'
                    right_test = 1
                elif pitch > 0.08:
                    direction = 'Looking Down'
                    down_test = 1
                elif pitch < -0.08:
                    direction = 'Looking Up'
                    up_test = 1
                
                cv2.putText(frame, f"Yaw: {yaw: .2f}", (30, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Pitch: {pitch: .2f}", (30, 90), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll: .2f}", (30, 120), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Direction: {direction}", (30, 160), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                if left_test == 1 and right_test == 1 & down_test == 1 and up_test == 1:
                    cv2.putText(frame, "Head Turn Test Passed", (60, 210), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 4)
                    head_turn_test = 1
                    break
        cv2.imshow("Head Rotation Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('9'):
            print('Terminating video stream . . .')
            break

    cap.release()
    cv2.destroyAllWindows()
    return head_turn_test


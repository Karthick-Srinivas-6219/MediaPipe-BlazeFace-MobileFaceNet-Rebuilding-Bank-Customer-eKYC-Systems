# importing dependencies

import cv2
import mediapipe
from math import sqrt
import numpy

# config params

counter = 0
total_blinks = 0
font = cv2.FONT_HERSHEY_COMPLEX

# mediapipe generated facial landmarks for both eyes

left_eye = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
right_eye = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

# importing some pre-trained solutions form mediapipe

face_mesh_processor = mediapipe.solutions.face_mesh # import face mesh processor 

# create a domain specific instance of the face mesh processor
mesh_applier = face_mesh_processor.FaceMesh(max_num_faces = 18, # single person use case
                                            min_detection_confidence = 0.6, # t.hold for detection
                                            min_tracking_confidence = 0.7) # t.hold for across frame tracking

def euclidean_distance(point1, point2):
    x, y = point1
    x1, y1 = point2
    distance = sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def get_landmarks(image, results, draw = False):
    image_height, image_width = image.shape[:2]
    mesh_coords = [(int(point.x * image_width), int(point.y * image_height)) for point in results.multi_face_landmarks[0].landmark]
    # iterates through each landmark and multiplies it with height & width to match resolution
    return mesh_coords

def blink_ratio(image, landmarks, right_indices, left_indices):
    
    # store co-ordinates of left and right tips of left and right eyes
    left_eye_left_tip = landmarks[left_indices[0]]
    left_eye_right_tip = landmarks[left_indices[8]]
    
    right_eye_left_tip = landmarks[right_indices[0]]
    right_eye_right_tip = landmarks[right_indices[8]]

    # store co-ordinates of the top and bottom tips of the left and right eyes
    left_eye_top_tip = landmarks[left_indices[12]]
    left_eye_bottom_tip = landmarks[left_indices[4]]

    right_eye_top_tip = landmarks[right_indices[12]]
    right_eye_bottom_tip = landmarks[right_indices[4]]

    # calculate horiziontal and vertical euclidean distances between eye tips for left and right eyes
    left_eye_hd = euclidean_distance(left_eye_left_tip, left_eye_right_tip)
    left_eye_vd = euclidean_distance(left_eye_top_tip, left_eye_bottom_tip)

    right_eye_hd = euclidean_distance(right_eye_left_tip, right_eye_right_tip)
    right_eye_vd = euclidean_distance(right_eye_top_tip, right_eye_bottom_tip)

    # calculate EAR for left and right eyes
    right_ear = right_eye_hd / right_eye_vd
    left_ear = left_eye_hd / left_eye_vd

    total_ear = (left_ear + right_ear)/2
    return total_ear

def count_blinks():
    # create a connection to webcam (indexed 0)
    video_capture = cv2.VideoCapture(0) 
    counter = 0
    total_blinks = 0
    while True:
        ret, frame = video_capture.read() # ret - bool success flag | frame - frame ad a 3D RGB numpy tensor
        if not ret:
            print('Failed to grab frame!')
            break
        frame = cv2.resize(frame, None, fx = 1.5, fy = 1.5, interpolation = cv2.INTER_CUBIC)
        # scales up the frame by 1.5x for both height & width and applies cubic interpolation to enlarge the image
        # cubic: preserves fine detail, ideal for face processing
        
        
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # opens the frame in RGB
        results = mesh_applier.process(rgb_frame) # detects a face and returns 468 3D landmarks corresponding to different facial features
        
        if results.multi_face_landmarks: # check for more than 1 face
            num_faces = len(results.multi_face_landmarks)
            if num_faces > 1:
                print('More than 1 face detected !')
                break
            mesh_coords = get_landmarks(frame, results, True)
            eyes_ratio = blink_ratio(frame, mesh_coords, right_eye, left_eye)
            cv2.putText(frame, "Please blink your eyes",(int(frame_height/2), 100), font, 1, (0, 255, 0), 2)
            if eyes_ratio > 3:
                counter +=1
            else:
                if counter > 4:
                    total_blinks +=1
                    if total_blinks > 5:
                        break
                    counter = 0
            cv2.rectangle(frame, (20, 120), (290, 260), (0,0,0), -1)
            cv2.putText(frame, f'total_blinks: {total_blinks}', (30, 150), font, 1, (0, 255, 0), 2)

            
        cv2.imshow('Video Stream', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('9'):
            print('Terminating Video Stream . . .')
            break
    video_capture.release() # close camera connection
    cv2.destroyAllWindows() # closes window
    if total_blinks > 5:
        return 1


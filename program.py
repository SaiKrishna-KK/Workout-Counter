# through terminal pip install mediapipe and opencv-python
# importing the dependencies
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils #selection of drawing functions to draw on the video for training
mp_pose = mp.solutions.pose #selection of pose estimation object from solution
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence= 0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened(): 
        ret, frame = cap.read()
        # Converting the image from bgr to rgb to read and write and later locking the image to not have any further writes
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Making pose estimations
        results= pose.process(image)

        #rgb to bgr and writeable = true
        image.flags.writeable= True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rendering the changes
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color = (245,117,66), thickness =2, circle_radius =2), 
                                mp_drawing.DrawingSpec(color = (245,66,230), thickness =2, circle_radius =2)
                                )

        cv2.imshow("Mediapipe Feed", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
 
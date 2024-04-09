import cv2
import mediapipe as mp
import numpy as np

# Import a function to calculate angles between points (replace with your actual module)
from exercise_models.calculate_angle import calculate_angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize exercise stage
stage = "starting"

print("Press 'q' to quit the window!!")

# Initialize the MediaPipe Pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect pose landmarks
        results = pose.process(image)

        try:
            landmarks = results.pose_landmarks.landmark

            # Extract relevant keypoints for plank
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angles
            angle_between_shoulders_hips_ankles = calculate_angle(left_shoulder, left_hip, left_ankle) + calculate_angle(right_shoulder, right_hip, right_ankle)

            # Check exercise stage based on angles
            if angle_between_shoulders_hips_ankles < 170:
                stage = "correct_plank"
            else:
                stage = "not_plank"

            # Display exercise stage
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        except:
            pass

        # Convert back to BGR format
        image.flags.writeable = Trueq
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Plank Monitoring', image)

        # Check for 'q' key press to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

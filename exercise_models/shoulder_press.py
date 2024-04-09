import cv2
import mediapipe as mp
import numpy as np
from exercise_models.calculate_angle import calculate_angle


import tkinter as tk

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to update the metrics in the tkinter window
def update_metrics():
    if 'right_shoulder_elbow_angle' in globals() and 'left_shoulder_elbow_angle' in globals() and 'Form' in globals() and 'stage' in globals():
        metrics_label.config(text=f'Right Shoulder-Elbow Angle: {right_shoulder_elbow_angle - elbow_angle_threshold:.2f} degrees\n'
                                  f'Left Shoulder-Elbow Angle: {left_shoulder_elbow_angle - elbow_angle_threshold:.2f} degrees\n'
                                  f'Form Assessment: {Form}\n'
                                  f'Counter: {str(counter)}\n'
                                  f'Form Stage: {stage}')
    metrics_label.after(100, update_metrics)

# Create a tkinter window
root = tk.Tk()
root.title("Exercise Metrics")

# Create a label to display the metrics
metrics_label = tk.Label(root, text="", font=("Arial", 14))
metrics_label.pack()

# Initialize metrics
right_shoulder_elbow_angle = 0
left_shoulder_elbow_angle = 0
Form = "unknown"
stage = "unknown"
counter = 0

# Start the metric update function
update_metrics()

print("Press q to quit the window!!")

cap = cv2.VideoCapture(0)
rep_count = 0  # Number of repetitions
set_count = 0  # Number of sets
rep_started = False  # Flag to track if a repetition has started
stage = "down"
Form = "wrong"
counter = 0  # Initialize the counter

start = False
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # detect
        results = pose.process(image)

        try:
            landmarks = results.pose_landmarks.landmark
            counter = 0

            # Define keypoints for shoulder press (e.g., right shoulder, right elbow, left shoulder, left elbow)
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Calculate the relevant angles (shoulder-elbow angle for each arm)
            right_shoulder_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_shoulder_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            RSH_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
            LSH_angle = calculate_angle(left_elbow, left_shoulder, left_hip)

            elbow_angle_threshold = 90.0  # Adjust as needed
            shoulder_angle_threshold = 90.0  # Adjust as needed
            tolerance = 5.0

            # # Determine if each arm's form is proper or not based on the angle thresholds
            right_arm_proper_form = bool(
                abs(right_shoulder_elbow_angle - elbow_angle_threshold) <= 80.0
                and abs(right_shoulder_elbow_angle - elbow_angle_threshold) > -10.0
            )
            left_arm_proper_form = bool(
                abs(right_shoulder_elbow_angle - elbow_angle_threshold) <= 80.0
                and abs(right_shoulder_elbow_angle - elbow_angle_threshold) > -10.0
            )

            # Assess overall form based on both arms
            if (right_arm_proper_form and left_arm_proper_form) and (RSH_angle >= 60 and LSH_angle >= 60):
                Form = "right"
                if stage == "down" and abs(right_shoulder_elbow_angle - elbow_angle_threshold) >= 60.0:

                    counter += 1  # Increment the counter only when a new repetition starts
                    # rep_started = True  # Set the flag to indicate that a repetition has started
                    stage = "up"
                elif stage == "up" and abs(right_shoulder_elbow_angle - elbow_angle_threshold) < 70.0:
                    stage = "down"
                    # rep_started = False  # Reset the flag when the arm is lowered
            else:
                Form = "wrong"

            # if  abs(right_shoulder_elbow_angle - elbow_angle_threshold) <3.0 and  abs(right_shoulder_elbow_angle - elbow_angle_threshold) < 40.0:
            #     stage = "right"
            # else:
            #     stage = "wrong"

            # cv2.putText(frame, f'Right Shoulder-Elbow Angle: {right_shoulder_elbow_angle:.2f} degrees', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(frame, f'Left Shoulder-Elbow Angle: {left_shoulder_elbow_angle:.2f} degrees', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # # cv2.putText(frame, f'Model Prediction: {prediction[0]:.2f}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(frame, f'Form Assessment: {stage}', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)




        except:
            pass

        # Stage data
        cv2.putText(image,
                    f'Right Shoulder-Elbow Angle: {right_shoulder_elbow_angle - elbow_angle_threshold:.2f} degrees',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image,
                    f'Left Shoulder-Elbow Angle: {left_shoulder_elbow_angle - elbow_angle_threshold:.2f} degrees',
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, f'Model Prediction: {prediction[0]:.2f}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Form Assessment: {Form}', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, f'Counter:{str(counter)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, f'Form Assessment: {stage}', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        # to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# Start the tkinter mainloop
root.mainloop()
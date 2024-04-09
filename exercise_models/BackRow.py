import cv2
import mediapipe as mp
import math

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    angle_rad = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle_rad = angle_rad % (2 * math.pi)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# Function to check if the angle is within the desired range
def is_correct_pose(angle, lower_bound, upper_bound):
    return lower_bound <= angle <= upper_bound

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        # Extract landmarks for left shoulder, left elbow, and left wrist
        left_shoulder = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
                         int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]))

        left_elbow = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1]),
                      int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0]))

        left_wrist = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1]),
                      int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0]))

        # Extract landmarks for left hip and left knee
        left_hip = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * frame.shape[1]),
                    int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * frame.shape[0]))

        left_knee = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * frame.shape[1]),
                     int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * frame.shape[0]))

        # Calculate the angles
        angle_shoulder_elbow_wrist = calculate_angle(left_shoulder, left_elbow, left_wrist)
        angle_hip_knee_shoulder = calculate_angle(left_hip, left_knee, left_shoulder)

        # Check if the angles are within the desired ranges
        if is_correct_pose(angle_shoulder_elbow_wrist, 190, 275) and is_correct_pose(angle_hip_knee_shoulder, 325, 332):
            message = f'Correct Pose! Shoulder-Elbow-Wrist Angle: {round(angle_shoulder_elbow_wrist, 2)} degrees, Hip-Knee-Shoulder Angle: {round(angle_hip_knee_shoulder, 2)} degrees'
        else:
            message = f'Incorrect Pose! Shoulder-Elbow-Wrist Angle: {round(angle_shoulder_elbow_wrist, 2)} degrees, Hip-Knee-Shoulder Angle: {round(angle_hip_knee_shoulder, 2)} degrees'

        # Draw circles at the landmark positions
        cv2.circle(frame, left_shoulder, 5, (0, 255, 0), -1)
        cv2.circle(frame, left_elbow, 5, (0, 255, 0), -1)
        cv2.circle(frame, left_wrist, 5, (0, 255, 0), -1)
        cv2.circle(frame, left_hip, 5, (0, 255, 0), -1)
        cv2.circle(frame, left_knee, 5, (0, 255, 0), -1)

        # Display the message on the screen
        cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Angle Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

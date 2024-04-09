import cv2
import mediapipe as mp
import math

def calculate_angle(a, b, c):
    angle_radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle_degrees = math.degrees(angle_radians)
    return abs(angle_degrees)

def count_reps(angle, lower_threshold, upper_threshold, in_down_phase, rep_count):
    if angle < lower_threshold and not in_down_phase:
        # Entering the down phase
        in_down_phase = True
    elif angle > upper_threshold and in_down_phase:
        # Entering the up phase and counting a repetition
        in_down_phase = False
        rep_count += 1
        print(f"Rep Count: {rep_count}")
    return in_down_phase, rep_count

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

right_shoulder_keypoint = 11
right_elbow_keypoint = 13
right_wrist_keypoint = 15

in_down_phase = False
rep_count = 0
lower_threshold = 40
upper_threshold = 150

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        right_shoulder = landmarks[right_shoulder_keypoint]
        right_elbow = landmarks[right_elbow_keypoint]
        right_wrist = landmarks[right_wrist_keypoint]

        if right_shoulder and right_elbow and right_wrist:
            angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            for landmark in [right_shoulder, right_elbow, right_wrist]:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            cv2.putText(frame, f"Angle: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            in_down_phase, rep_count = count_reps(angle, lower_threshold, upper_threshold, in_down_phase, rep_count)

    cv2.putText(frame, f"Rep Count: {rep_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Tricep Extension Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

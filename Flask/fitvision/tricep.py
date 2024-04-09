import cv2
import mediapipe as mp
import math


def calculate_angle(a, b, c):
    # Calculate the angle between three points (in radians)
    angle_radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle_degrees = math.degrees(angle_radians)
    return abs(angle_degrees)


def generate_frames():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize Video Capture
    cap = cv2.VideoCapture(0)  # Use your desired camera, 0 for default camera

    # Define key points for the tricep extension exercise
    right_shoulder_keypoint = 11
    right_elbow_keypoint = 13
    right_wrist_keypoint = 15

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a more intuitive view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose detection
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            right_shoulder = landmarks[right_shoulder_keypoint]
            right_elbow = landmarks[right_elbow_keypoint]
            right_wrist = landmarks[right_wrist_keypoint]

            # Calculate the angle between shoulder, elbow, and wrist
            if right_shoulder and right_elbow and right_wrist:
                angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Draw key points
                for landmark in [right_shoulder, right_elbow, right_wrist]:
                    x, y = int(landmark.x * frame.shape[1]), int(
                        landmark.y * frame.shape[0]
                    )
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                # Annotate the angle value
                cv2.putText(
                    frame,
                    f"Angle: {angle:.2f} degrees",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # You can set a threshold for what is considered a proper tricep extension
                upper_threshold = 150
                lower_threshold = 40

                # Determine if the exercise is being performed properly
                if angle > lower_threshold and angle < upper_threshold:
                    cv2.putText(
                        frame,
                        "Proper Tricep Extension",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "Improve Tricep Extension",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

        cv2.imshow("Tricep Extension Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

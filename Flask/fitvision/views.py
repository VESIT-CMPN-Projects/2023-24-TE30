from datetime import datetime
from flask import render_template, request, Response
from fitvision import app
import cv2
import mediapipe as mp
import math
import numpy as np
import subprocess


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def tricep():
    def calculate_angle(a, b, c):
        # Calculate the angle between three points (in radians)
        angle_radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(
            a.y - b.y, a.x - b.x
        )
        angle_degrees = math.degrees(angle_radians)
        return abs(angle_degrees)

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


def biceup_curl():

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Create a file to store metrics
    metrics_file = open("metrics.txt", "w")

    cap = cv2.VideoCapture(0)
    stage = "down"
    print("Press q to quit the window!!")
    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # detect
            results = pose.process(image)

            try:
                landmarks = results.pose_landmarks.landmark

                shoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                ]
                elbow = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                ]
                wrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                ]
                Hip = [
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                ]
                EL = calculate_angle(shoulder, elbow, wrist)
                Sh = calculate_angle(elbow, shoulder, Hip)
                # puttext
                cv2.putText(
                    image,
                    str(EL),
                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    image,
                    str(Sh),
                    tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "STAGE",
                    (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    stage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Update metrics in the file
                metrics_file.seek(0)
                metrics_file.write(f"{EL}\n{Sh}\n{stage}\n")
                metrics_file.truncate()

                if (EL > 160 and Sh < 25) or (EL < 40 and stage and Sh < 25):
                    stage = "right"
                else:
                    stage = "wrong"

            except:
                pass

            # Stage data
            cv2.putText(
                image,
                "",
                (65, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                stage,
                (60, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            # to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # detection
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            cv2.imshow("Mediapipe Feed", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Close the metrics file
    metrics_file.close()


def backrow():

    # Function to calculate the angle between three points
    def calculate_angle(a, b, c):
        angle_rad = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(
            a[1] - b[1], a[0] - b[0]
        )
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
            left_shoulder = (
                int(
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ].x
                    * frame.shape[1]
                ),
                int(
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ].y
                    * frame.shape[0]
                ),
            )

            left_elbow = (
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x
                    * frame.shape[1]
                ),
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
                    * frame.shape[0]
                ),
            )

            left_wrist = (
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
                    * frame.shape[1]
                ),
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
                    * frame.shape[0]
                ),
            )

            # Extract landmarks for left hip and left knee
            left_hip = (
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x
                    * frame.shape[1]
                ),
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
                    * frame.shape[0]
                ),
            )

            left_knee = (
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x
                    * frame.shape[1]
                ),
                int(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
                    * frame.shape[0]
                ),
            )

            # Calculate the angles
            angle_shoulder_elbow_wrist = calculate_angle(
                left_shoulder, left_elbow, left_wrist
            )
            angle_hip_knee_shoulder = calculate_angle(
                left_hip, left_knee, left_shoulder
            )

            # Check if the angles are within the desired ranges
            if is_correct_pose(
                angle_shoulder_elbow_wrist, 190, 275
            ) and is_correct_pose(angle_hip_knee_shoulder, 325, 332):
                message = f"Correct Pose! Shoulder-Elbow-Wrist Angle: {round(angle_shoulder_elbow_wrist, 2)} degrees, Hip-Knee-Shoulder Angle: {round(angle_hip_knee_shoulder, 2)} degrees"
            else:
                message = f"Incorrect Pose! Shoulder-Elbow-Wrist Angle: {round(angle_shoulder_elbow_wrist, 2)} degrees, Hip-Knee-Shoulder Angle: {round(angle_hip_knee_shoulder, 2)} degrees"

            # Draw circles at the landmark positions
            cv2.circle(frame, left_shoulder, 5, (0, 255, 0), -1)
            cv2.circle(frame, left_elbow, 5, (0, 255, 0), -1)
            cv2.circle(frame, left_wrist, 5, (0, 255, 0), -1)
            cv2.circle(frame, left_hip, 5, (0, 255, 0), -1)
            cv2.circle(frame, left_knee, 5, (0, 255, 0), -1)

            # Display the message on the screen
            cv2.putText(
                frame,
                message,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Display the frame
        cv2.imshow("Angle Detection", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/tricep")
def tricep_path():

    return Response(tricep(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/biceup_curl")
def biceup_path():
    return Response(biceup_curl(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/back_row")
def back_row():
    return Response(backrow(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/home")
def home():
    """Renders the home page."""
    return render_template(
        "index.html",
        title="Home Page",
        year=datetime.now().year,
    )


@app.route("/about")
def about():
    """Renders the about page."""
    return render_template(
        "about.html",
        title="About",
        year=datetime.now().year,
        message="Your application description page.",
    )


@app.route("/explore")
def explore():
    """Renders the explore page."""
    # Add logic to fetch and display exercises
    return render_template(
        "explore.html",
        title="Explore Exercises",
        year=datetime.now().year,
    )


@app.route("/login")
def login():
    """Renders the login page."""
    return render_template(
        "login.html",
        title="Login",
        year=datetime.now().year,
    )


@app.route("/user_login/book_session")
def book_session():
    """Renders the book session page for users."""
    # Add logic to handle session bookings
    return render_template(
        "book_session.html",
        title="Book a Session",
        year=datetime.now().year,
    )


@app.route("/trainer_login/session_requests")
def session_requests():
    """Renders the session requests page for trainers."""
    # Add logic to handle session requests
    # Dummy session request entries
    session_request1 = {
        "user_name": "User 1",
        "user_email": "user1@example.com",
        "user_phone": "123-456-7890",
        "session_type": "Personal Training",
        "preferred_trainer": "Trainer A",
        "session_date": "2024-03-10",
        "session_time": "10:00 AM",
    }

    session_request2 = {
        "user_name": "User 2",
        "user_email": "user2@example.com",
        "user_phone": "987-654-3210",
        "session_type": "Group Training",
        "preferred_trainer": "Trainer B",
        "session_date": "2024-03-12",
        "session_time": "02:00 PM",
    }

    # Create a list of session requests
    session_requests_data = [session_request1, session_request2]

    return render_template(
        "session_requests.html", session_requests=session_requests_data
    )


# @app.route('/book_session', methods=['GET', 'POST'])
# def book_session():
#     if request.method == 'POST':
#         full_name = request.form.get('fullName')
#         email = request.form.get('email')
#         phone = request.form.get('phone')
#         session_type = request.form.get('sessionType')
#         preferred_trainer = request.form.get('preferredTrainer')
#         session_date = request.form.get('sessionDate')
#         # Add your session booking logic here


#     return render_template('book_session.html')


from flask import render_template
from flask.views import MethodView


class CameraView(MethodView):
    def get(self):
        return render_template("camera.html")


# Add the route for the CameraView
app.add_url_rule("/camera", view_func=CameraView.as_view("camera"))

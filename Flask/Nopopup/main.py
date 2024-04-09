from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math

app = Flask(__name__)

frame_width = 640
frame_height = 480
frame_rate = 60

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize VideoCapture object
cap = cv2.VideoCapture(0)

def bicep():
    def calculate_angle(a, b, c):
        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = math.degrees(radians)
        angle = (angle + 360) % 360  # Convert angle to be between 0 and 360
        angle = min(angle, 360 - angle)  # Get the smaller angle between angle and 360 - angle
        return angle

    def detect_bicep(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = {lmk: lm for lmk, lm in enumerate(results.pose_landmarks.landmark)}

            # Calculate angles for left arm
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # Draw feedback on frame
            min_angle = 45
            max_angle = 170
            cv2.putText(frame, f"Left Arm Angle: {round(left_arm_angle, 2)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

            # Check if the left arm angle falls within the desired range
            if min_angle <= left_arm_angle <= max_angle:
                cv2.putText(frame, "Bicep Curl Correct", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Incorrect Form", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            relevant_landmarks = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                                  mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                                  mp.solutions.pose.PoseLandmark.LEFT_WRIST
                                  ]

            for landmark in relevant_landmarks:
                landmark_point = results.pose_landmarks.landmark[landmark.value]
                h, w, c = frame.shape
                cx, cy = int(landmark_point.x * w), int(landmark_point.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Draw a circle f

        return frame

    return calculate_angle, detect_bicep

def generate_frames(detect_bicep):
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_bicep(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exercise1')
def exercise1():
    # Run the bicep function to set up for exercise 1
    calculate_angle, detect_bicep = bicep()
    global generate_frames_exercise1
    generate_frames_exercise1 = generate_frames(detect_bicep)
    return render_template('exercise1.html')

@app.route('/exercise2')
def exercise2():
    return render_template('exercise2.html')

@app.route('/video_feed')
def video_feed():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)
    return Response(generate_frames_exercise1, mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

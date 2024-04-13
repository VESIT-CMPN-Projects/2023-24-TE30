from datetime import datetime
from flask import render_template, request, Response, redirect, session
from fitvision import app
import cv2
import mediapipe as mp
import math
import numpy as np
import subprocess
import pyrebase

app.secret_key = "your_secret_key"

firebase_config = {
    "apiKey": "AIzaSyAL3E_w1gaR_qaXIw-etd1vVQYbBTctPKs",
    "authDomain": "evac-77db5.firebaseapp.com",
    "databaseURL": "https://evac-77db5-default-rtdb.firebaseio.com/",
    "projectId": "evac-77db5",
    "storageBucket": "evac-77db5.appspot.com",
    "messagingSenderId": "832522615412",
    "appId": "1:832522615412:web:57d363bc5a2422fb816f21",
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()


frame_width = 640
frame_height = 480
frame_rate = 60

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = None  # Initialize VideoCapture object globally


def bicep():
    def calculate_angle(a, b, c):
        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = math.degrees(radians)
        angle = (angle + 360) % 360  # Convert angle to be between 0 and 360
        angle = min(
            angle, 360 - angle
        )  # Get the smaller angle between angle and 360 - angle
        return angle

    def detect_bicep(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = {
                lmk: lm for lmk, lm in enumerate(results.pose_landmarks.landmark)
            }

            left_shoulder = landmarks[
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
            ]
            left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            min_angle = 45
            max_angle = 170
            cv2.putText(
                frame,
                f"Left Arm Angle: {round(left_arm_angle, 2)}",
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            if min_angle <= left_arm_angle <= max_angle:
                cv2.putText(
                    frame,
                    "Bicep Curl Correct",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Incorrect Form",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            relevant_landmarks = [
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            ]

            for landmark in relevant_landmarks:
                landmark_point = results.pose_landmarks.landmark[landmark.value]
                h, w, c = frame.shape
                cx, cy = int(landmark_point.x * w), int(landmark_point.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        return frame

    return calculate_angle, detect_bicep


def backrow():
    def calculate_angle(a, b, c):
        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = math.degrees(radians)
        angle = (angle + 360) % 360  # Convert angle to be between 0 and 360
        angle = min(
            angle, 360 - angle
        )  # Get the smaller angle between angle and 360 - angle
        return angle

    def is_correct_pose(angle, lower_bound, upper_bound):
        return lower_bound <= angle <= upper_bound

    def detect_bicep(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = {
                lmk: lm for lmk, lm in enumerate(results.pose_landmarks.landmark)
            }

            left_shoulder = landmarks[
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
            ]
            left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]

            angle_shoulder_elbow_wrist = calculate_angle(
                left_shoulder, left_elbow, left_wrist
            )
            angle_hip_knee_shoulder = calculate_angle(
                left_hip, left_knee, left_shoulder
            )

            cv2.putText(
                frame,
                f"angle_shoulder_elbow_wrist {round(angle_shoulder_elbow_wrist, 2)}",
                (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Left Arm Angle: {round(angle_hip_knee_shoulder, 2)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            if is_correct_pose(angle_shoulder_elbow_wrist, 70, 180) and is_correct_pose(
                angle_hip_knee_shoulder, 20, 35
            ):
                # Code block if left_arm_angle is outside the range
                cv2.putText(
                    frame,
                    "BackRow Correct",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Incorrect Form",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            relevant_landmarks = [
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                mp.solutions.pose.PoseLandmark.LEFT_HIP,
                mp.solutions.pose.PoseLandmark.LEFT_KNEE,
            ]

            for landmark in relevant_landmarks:
                landmark_point = results.pose_landmarks.landmark[landmark.value]
                h, w, c = frame.shape
                cx, cy = int(landmark_point.x * w), int(landmark_point.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        return frame

    return calculate_angle, detect_bicep


def shoulderpress():
    def calculate_angle(a, b, c):
        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = math.degrees(radians)
        angle = (angle + 360) % 360  # Convert angle to be between 0 and 360
        angle = min(
            angle, 360 - angle
        )  # Get the smaller angle between angle and 360 - angle
        return angle

    def is_correct_pose(angle, lower_bound, upper_bound):
        return lower_bound <= angle <= upper_bound

    def detect_bicep(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = {
                lmk: lm for lmk, lm in enumerate(results.pose_landmarks.landmark)
            }

            left_shoulder = landmarks[
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
            ]
            left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]

            right_shoulder = landmarks[
                mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
            ]
            right_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

            angle_lshoulder_lelbow_lwrist = calculate_angle(
                left_shoulder, left_elbow, left_wrist
            )
            angle_rshoulder_relbow_rwrist = calculate_angle(
                right_shoulder, right_elbow, right_wrist
            )
            angle_lelbow_lshoulder_lhip = calculate_angle(
                left_elbow, left_shoulder, left_hip
            )
            angle_relbow_rshoulder_rhip = calculate_angle(
                right_elbow, right_shoulder, right_hip
            )

            cv2.putText(
                frame,
                f"angle_shoulder_elbow_wrist {round(angle_lelbow_lshoulder_lhip, 2)}",
                (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            if (
                is_correct_pose(angle_lshoulder_lelbow_lwrist, 60, 170)
                and is_correct_pose(angle_rshoulder_relbow_rwrist, 60, 170)
                and is_correct_pose(angle_lelbow_lshoulder_lhip, 70, 180)
                and is_correct_pose(angle_relbow_rshoulder_rhip, 70, 180)
            ):
                # Code block if left_arm_angle is outside the range
                cv2.putText(
                    frame,
                    "shoulderpress Correct",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Incorrect Form",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            relevant_landmarks = [
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
                mp.solutions.pose.PoseLandmark.LEFT_HIP,
                mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            ]

            for landmark in relevant_landmarks:
                landmark_point = results.pose_landmarks.landmark[landmark.value]
                h, w, c = frame.shape
                cx, cy = int(landmark_point.x * w), int(landmark_point.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        return frame

    return calculate_angle, detect_bicep


def plank():
    def calculate_angle(a, b, c):
        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = math.degrees(radians)
        angle = (angle + 360) % 360  # Convert angle to be between 0 and 360
        angle = min(
            angle, 360 - angle
        )  # Get the smaller angle between angle and 360 - angle
        return angle

    def is_correct_pose(angle, lower_bound, upper_bound):
        return lower_bound <= angle <= upper_bound

    def detect_bicep(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = {
                lmk: lm for lmk, lm in enumerate(results.pose_landmarks.landmark)
            }

            left_shoulder = landmarks[
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
            ]
            left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]

            right_shoulder = landmarks[
                mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
            ]
            right_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

            angle_lshoulder_lelbow_lwrist = calculate_angle(
                left_shoulder, left_elbow, left_wrist
            )
            angle_rshoulder_relbow_rwrist = calculate_angle(
                right_shoulder, right_elbow, right_wrist
            )
            angle_lelbow_lshoulder_lhip = calculate_angle(
                left_elbow, left_shoulder, left_hip
            )
            angle_relbow_rshoulder_rhip = calculate_angle(
                right_elbow, right_shoulder, right_hip
            )

            cv2.putText(
                frame,
                f"angle_shoulder_elbow_wrist {round(angle_lelbow_lshoulder_lhip, 2)}",
                (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            if (
                is_correct_pose(angle_lshoulder_lelbow_lwrist, 60, 170)
                and is_correct_pose(angle_rshoulder_relbow_rwrist, 60, 170)
                and is_correct_pose(angle_lelbow_lshoulder_lhip, 70, 180)
                and is_correct_pose(angle_relbow_rshoulder_rhip, 70, 180)
            ):
                # Code block if left_arm_angle is outside the range
                cv2.putText(
                    frame,
                    "shoulderpress Correct",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Incorrect Form",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            relevant_landmarks = [
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
                mp.solutions.pose.PoseLandmark.LEFT_HIP,
                mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            ]

            for landmark in relevant_landmarks:
                landmark_point = results.pose_landmarks.landmark[landmark.value]
                h, w, c = frame.shape
                cx, cy = int(landmark_point.x * w), int(landmark_point.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        return frame

    return calculate_angle, detect_bicep


def lunges():
    def calculate_angle(a, b, c):
        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = math.degrees(radians)
        angle = (angle + 360) % 360  # Convert angle to be between 0 and 360
        angle = min(
            angle, 360 - angle
        )  # Get the smaller angle between angle and 360 - angle
        return angle

    def is_correct_pose(angle, lower_bound, upper_bound):
        return lower_bound <= angle <= upper_bound

    def detect_bicep(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = {
                lmk: lm for lmk, lm in enumerate(results.pose_landmarks.landmark)
            }

            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]

            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]

            angle_lhip_lknee_lankle = calculate_angle(left_hip, left_knee, left_ankle)
            angle_rhip_rknee_rankle = calculate_angle(
                right_hip, right_knee, right_ankle
            )

            cv2.putText(
                frame,
                f"angle_shoulder_elbow_wrist {round(angle_rhip_rknee_rankle, 2)}",
                (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            if is_correct_pose(angle_lhip_lknee_lankle, 70, 170) and is_correct_pose(
                angle_rhip_rknee_rankle, 100, 170
            ):
                # Code block if left_arm_angle is outside the range
                cv2.putText(
                    frame,
                    "shoulderpress Correct",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Incorrect Form",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            relevant_landmarks = [
                mp.solutions.pose.PoseLandmark.LEFT_HIP,
                mp.solutions.pose.PoseLandmark.LEFT_KNEE,
                mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
                mp.solutions.pose.PoseLandmark.RIGHT_HIP,
                mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
                mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
            ]

            for landmark in relevant_landmarks:
                landmark_point = results.pose_landmarks.landmark[landmark.value]
                h, w, c = frame.shape
                cx, cy = int(landmark_point.x * w), int(landmark_point.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        return frame

    return calculate_angle, detect_bicep


def lllift():
    def calculate_angle(a, b, c):
        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = math.degrees(radians)
        angle = (angle + 360) % 360  # Convert angle to be between 0 and 360
        angle = min(
            angle, 360 - angle
        )  # Get the smaller angle between angle and 360 - angle
        return angle

    def is_correct_pose(angle, lower_bound, upper_bound):
        return lower_bound <= angle <= upper_bound

    def detect_bicep(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = {
                lmk: lm for lmk, lm in enumerate(results.pose_landmarks.landmark)
            }

            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
            right_ankle = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]

            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
            left_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]

            angle_lhip_lknee_lankle = calculate_angle(left_hip, left_knee, right_ankle)
            angle_rhip_rknee_rankle = calculate_angle(right_hip, right_knee, left_ankle)

            cv2.putText(
                frame,
                f"angle_lhip_lknee_lankle {round(angle_lhip_lknee_lankle, 2)}",
                (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            if is_correct_pose(angle_lhip_lknee_lankle, 160, 180) or is_correct_pose(
                angle_rhip_rknee_rankle, 160, 180
            ):
                # Code block if left_arm_angle is outside the range
                cv2.putText(
                    frame,
                    "shoulderpress Correct",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Incorrect Form",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            relevant_landmarks = [
                mp.solutions.pose.PoseLandmark.LEFT_HIP,
                mp.solutions.pose.PoseLandmark.LEFT_KNEE,
                mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
                mp.solutions.pose.PoseLandmark.RIGHT_HIP,
                mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
                mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
            ]

            for landmark in relevant_landmarks:
                landmark_point = results.pose_landmarks.landmark[landmark.value]
                h, w, c = frame.shape
                cx, cy = int(landmark_point.x * w), int(landmark_point.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        return frame

    return calculate_angle, detect_bicep


def tricepextension():
    def calculate_angle(a, b, c):
        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = math.degrees(radians)
        angle = (angle + 360) % 360  # Convert angle to be between 0 and 360
        angle = min(
            angle, 360 - angle
        )  # Get the smaller angle between angle and 360 - angle
        return angle

    def is_correct_pose(angle, lower_bound, upper_bound):
        return lower_bound <= angle <= upper_bound

    def detect_bicep(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = {
                lmk: lm for lmk, lm in enumerate(results.pose_landmarks.landmark)
            }

            left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
            left_shoulder = landmarks[
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
            ]
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]

            right_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
            right_shoulder = landmarks[
                mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
            ]
            right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

            angle_lshoulder_lelbow_lwrist = calculate_angle(
                left_shoulder, left_elbow, left_wrist
            )
            angle_rshoulder_relbow_rwrist = calculate_angle(
                right_shoulder, right_elbow, right_wrist
            )
            angle_lelbow_lshoulder_lhip = calculate_angle(
                left_elbow, left_shoulder, left_hip
            )
            angle_relbow_rshoulder_rhip = calculate_angle(
                right_elbow, right_shoulder, right_hip
            )

            cv2.putText(
                frame,
                f"angle_shoulder_elbow_wrist {round(angle_lshoulder_lelbow_lwrist, 2)}",
                (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            if (
                is_correct_pose(angle_lshoulder_lelbow_lwrist, 20, 170)
                and is_correct_pose(angle_rshoulder_relbow_rwrist, 20, 170)
                and is_correct_pose(angle_lelbow_lshoulder_lhip, 170, 180)
                and is_correct_pose(angle_relbow_rshoulder_rhip, 170, 180)
            ):
                # Code block if left_arm_angle is outside the range
                cv2.putText(
                    frame,
                    "shoulderpress Correct",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Incorrect Form",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            relevant_landmarks = [
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
                mp.solutions.pose.PoseLandmark.LEFT_HIP,
                mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            ]

            for landmark in relevant_landmarks:
                landmark_point = results.pose_landmarks.landmark[landmark.value]
                h, w, c = frame.shape
                cx, cy = int(landmark_point.x * w), int(landmark_point.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        return frame

    return calculate_angle, detect_bicep


def generate_frames(detect_bicep):
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_bicep(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )


@app.route("/", methods=["GET", "POST"])
def index():
    global cap
    if request.method == "POST":
        # Release the VideoCapture object
        email = request.form["userEmail"]
        password = request.form["userPassword"]
        print("There is no p here")
        try:
            # Authenticate user with Pyrebase
            user = auth.sign_in_with_email_and_password(email, password)
            session["uid"] = user["localId"]  # Save user ID in session
            print("There is no p1 here")
            return redirect("/home")
        except pyrebase.pyrebase.HTTPError:
            return "Invalid credentials. Please try again."

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["pass"]
        try:
            # Create a new user with Pyrebase
            user = auth.create_user_with_email_and_password(email, password)
            session["uid"] = user["localId"]  # Save user ID in session
            return redirect("/login")
        except pyrebase.pyrebase.HTTPError as e:
            error_message = str(e.args[1])
            return render_template("register.html", error=error_message)
    return render_template("register.html")


@app.route("/exercise1")  # bicep
def exercise1():
    global cap
    if cap is not None:
        cap.release()
    calculate_angle, detect_bicep = bicep()
    global generate_frames_exercise1
    generate_frames_exercise1 = generate_frames(detect_bicep)
    return render_template("exercise1.html")


@app.route("/exercise2")  # backrow
def exercise2():
    global cap
    if cap is not None:
        cap.release()
    calculate_angle, detect_bicep = backrow()
    global generate_frames_exercise1
    generate_frames_exercise1 = generate_frames(detect_bicep)
    return render_template("exercise2.html")


@app.route("/exercise3")  # shoulderpress
def exercise3():
    global cap
    if cap is not None:
        cap.release()
    calculate_angle, detect_bicep = shoulderpress()
    global generate_frames_exercise1
    generate_frames_exercise1 = generate_frames(detect_bicep)
    return render_template("exercise3.html")


@app.route("/exercise4")  # shoulderpress
def exercise4():
    global cap
    if cap is not None:
        cap.release()
    calculate_angle, detect_bicep = plank()
    global generate_frames_exercise1
    generate_frames_exercise1 = generate_frames(detect_bicep)
    return render_template("exercise4.html")


@app.route("/exercise5")  # shoulderpress
def exercise5():
    global cap
    if cap is not None:
        cap.release()
    calculate_angle, detect_bicep = lunges()
    global generate_frames_exercise1
    generate_frames_exercise1 = generate_frames(detect_bicep)
    return render_template("exercise5.html")


@app.route("/exercise6")  # shoulderpress
def exercise6():
    global cap
    if cap is not None:
        cap.release()
    calculate_angle, detect_bicep = lllift()
    global generate_frames_exercise1
    generate_frames_exercise1 = generate_frames(detect_bicep)
    return render_template("exercise6.html")


@app.route("/exercise7")  # shoulderpress
def exercise7():
    global cap
    if cap is not None:
        cap.release()
    calculate_angle, detect_bicep = tricepextension()
    global generate_frames_exercise1
    generate_frames_exercise1 = generate_frames(detect_bicep)
    return render_template("exercise7.html")


@app.route("/video_feed")
def video_feed():
    global cap
    cap = cv2.VideoCapture(0)  # Initialize VideoCapture object
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)
    return Response(
        generate_frames_exercise1, mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/home")
def home():
    """Renders the home page."""
    global cap
    if "uid" in session:

        return render_template(
            "index.html",
            title="Home Page",
            year=datetime.now().year,
        )
    else:
        return redirect("/login")


@app.route("/about")
def about():
    """Renders the about page."""
    global cap
    if cap is not None:
        cap.release()
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
    global cap
    if cap is not None:
        cap.release()
    return render_template(
        "explore.html",
        title="Explore Exercises",
        year=datetime.now().year,
    )


@app.route("/login")
def login():
    """Renders the login page."""
    global cap
    if cap is not None:
        cap.release()
    return render_template(
        "login.html",
        title="Login",
        year=datetime.now().year,
    )


@app.route("/user_login/book_session")
def book_session():
    """Renders the book session page for users."""
    # Add logic to handle session bookings
    global cap
    if cap is not None:
        cap.release()
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
    global cap
    if cap is not None:
        cap.release()
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

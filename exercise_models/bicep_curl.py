import cv2
import mediapipe as mp
import numpy as np
from exercise_models.calculate_angle import calculate_angle


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Create a file to store metrics
metrics_file = open("metrics.txt", "w")


cap = cv2.VideoCapture(0)
stage = "down"
print("Press q to quit the window!!")
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

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            Hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            EL = calculate_angle(shoulder, elbow, wrist)
            Sh = calculate_angle(elbow, shoulder, Hip)
            # puttext
            cv2.putText(image, str(EL),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            cv2.putText(image, str(Sh),
                        tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

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
        cv2.putText(image, '', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
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

# Close the metrics file
metrics_file.close()

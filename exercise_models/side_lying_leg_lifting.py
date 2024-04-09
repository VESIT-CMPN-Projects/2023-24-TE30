import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils    # Assigning drawing_utils from mediapipe as mp_drawing
mp_holistic = mp.solutions.holistic        # Assigning holistic from mediapipe as mp_holistic

def angle_between_lines(x1, y1, x2, y2, x3, y3):         # Defining a function to calculate angle between lines
    # Calculate the slopes of the two lines
    slope1 = (y2 - y1) / (x2 - x1)
    slope2 = (y3 - y2) / (x3 - x2)

    # Calculate the angle between the two lines
    angle = math.atan2(slope2 - slope1, 1 + slope1 * slope2)   # Calculate the angle using the slopes

    # Convert the angle to degrees and return it
    return math.degrees(angle)                                # Return the angle in degrees

leglift = 0           # Initialize a variable to count the number of leg lifts
count1 = False        # Initialize a boolean variable to keep track of the first position
count2 = False        # Initialize a boolean variable to keep track of the second position
count3 = False        # Initialize a boolean variable to keep track of the third position

# Initializing the Holistic model with minimum detection and tracking confidence
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    cap = cv2.VideoCapture('sidelyinglegliftvideo.mp4')    # Start capturing the video from the file "sidelyinglegliftvideo.mp4"
    # cap = cv2.VideoCapture(0)                            # Alternatively, we can capture the video from the webcam (0)

    while cap.isOpened():                # While the video is being captured
        ret, frame = cap.read()          # Read the frame from the video

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # Convert the image to RGB

        results = holistic.process(image)    # Make a detection using the Holistic model on the image

        annotated_image = image.copy()       # Make a copy of the image to draw landmarks on

        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)   # Draw the detected landmarks on the image

        left_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]     # Get the coordinates of the left hip
        right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]   # Get the coordinates of the right hip

        midpoint = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)    # Calculate the midpoint between the left hip and right hip

        left_knee = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE]    # Get the coordinates
        right_knee = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE]

        angle1 = angle_between_lines(left_knee.x, left_knee.y, midpoint[0], midpoint[1], right_knee.x, right_knee.y)
        print("Angles :",angle1)

        if (angle1 > 60):
            count1 = True
        if (count1 == True and angle1 > 100):
            count2 = True
        if (count2 == True and angle1 < 60):
            count3 = True
        if (count1 == True and count2 == True and count3 == True):
            leglift = leglift + 1
            count1 = False
            count2 = False
            count3 = False
        lg = leglift

        print("Leg Lift : ",leglift)
        # Draw a circle at the midpoint
        cv2.circle(annotated_image, (int(midpoint[0] * annotated_image.shape[1]), int(midpoint[1] * annotated_image.shape[0])), 5, (255, 0, 0), -1)

        # check if angle is between 68.85 to 80 and display "Correct Exercise" on screen
        if 68.85 <= angle1 <= 80:
            cv2.putText(annotated_image, "Correct Side Lying Leg Lift Exercise", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_image, "Incorrect Side Lying Leg Lift Exercise", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the value of angle1 and leglift on the output screen
        cv2.putText(annotated_image, "Angle: " + str(round(angle1, 2)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Display the value of angle1 and leglift on the output screen
        cv2.putText(annotated_image, "Leg Lift: " + str(round(lg, 2)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the annotated image
        cv2.imshow('MediaPipe Holistic', annotated_image)

        # Exit if the user presses the 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

def angle_between_lines(x1, y1, x2, y2, x3, y3):         # Defining a function to calculate angle between lines
    # Calculate the slopes of the two lines
    slope1 = (y2 - y1) / (x2 - x1)
    slope2 = (y3 - y2) / (x3 - x2)

    # Calculate the angle between the two lines
    angle = math.atan2(slope2 - slope1, 1 + slope1 * slope2)   # Calculate the angle using the slopes

    # Convert the angle to degrees and return it
    return math.degrees(angle)                                # Return the angle in degrees

class LegLiftDetector:
    def __init__(self, video_path, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.leg_lift_count = 0
        self.count1 = False
        self.count2 = False
        self.count3 = False

        self.cap = cv2.VideoCapture("sidelyinglegliftvideo.mp4")
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image)

            annotated_image = image.copy()
            self.mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

            left_hip = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_HIP]

            midpoint = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)

            left_knee = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_KNEE]
            right_knee = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_KNEE]

            angle1 = angle_between_lines(left_knee.x, left_knee.y, midpoint[0], midpoint[1], right_knee.x, right_knee.y)

            if (angle1 > 60):
                self.count1 = True
            if (self.count1 == True and angle1 > 100):
                self.count2 = True
            if (self.count2 == True and angle1 < 60):
                self.count3 = True
            if (self.count1 == True and self.count2 == True and self.count3 == True):
                self.leg_lift_count = self.leg_lift_count + 1
                self.count1 = False
                self.count2 = False
                self.count3 = False

            print("Leg Lift : ",self.leg_lift_count)

            cv2.circle(annotated_image, (int(midpoint[0] * annotated_image.shape[1]), int(midpoint[1] * annotated_image.shape[0])), 5, (255, 0, 0), -1)

            if 68.85 <= angle1 <= 80:
                cv2.putText(annotated_image, "Correct Side Lying Leg Lift Exercise", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_image, "Incorrect Side Lying Leg Lift Exercise", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(annotated_image, "Angle: " + str(round(angle1, 2)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            #         # Display the value of angle1 and leglift on the output screen
            # cv2.putText(annotated_image, "Leg Lift: " + str(round(lg, 2)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(annotated_image, "Leg Lift: " + str(round(self.leg_lift_count, 2)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Show the annotated image
            cv2.imshow('MediaPipe Holistic', annotated_image)

            # Exit if the user presses the 'q' key
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Release the webcam and close the window
        cap.release()
        cv2.destroyAllWindows()

detector = LegLiftDetector('sidelyinglegliftvideo.mp4')
detector.run()



## Class Method with GUI
import cv2
import mediapipe as mp
import numpy as np
import math
import tkinter as tk
from PIL import Image, ImageTk
import threading
from tkinter import filedialog

class LegLiftDetectorGUI:
    def __init__(self, master, video_path, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.leg_lift_count = 0
        self.count1 = False
        self.count2 = False
        self.count3 = False

        self.cap = cv2.VideoCapture(video_path)
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

        self.master = master
        self.master.title("Side Lying Leg Lift Detector")

        self.canvas = tk.Canvas(self.master, width=640, height=480)
        self.canvas.pack()

        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(side=tk.LEFT)

        self.start_button = tk.Button(self.button_frame, text="Start", command=self.start_video, width=10, height=4, bg="lightgreen", font=("Helvetica", 12, "bold"))
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_video, width=10, height=4, bg="red", font=("Helvetica", 12, "bold"))
        self.stop_button.pack(side=tk.LEFT)

        self.upload_button = tk.Button(self.button_frame, text="Upload Video", command=self.upload_video, width=10, height=4, bg="green", font=("Arial", 12, "bold"))
        self.upload_button.pack(side=tk.LEFT)

        self.leg_lift_count_label = tk.Label(self.master, text="Leg Lift Count: 0", foreground="black",  font=("Helvetica", 12, "bold"))
        self.leg_lift_count_label.pack(side=tk.LEFT, anchor=tk.CENTER)

        self.angle_label = tk.Label(self.master, text="Angle: 0", foreground="blue", font=("Helvetica", 12, "bold"))
        self.angle_label.pack(side=tk.BOTTOM, anchor=tk.CENTER)

        self.leg_lift_image = None
        self.video_running = False
        self.update_video()

    def upload_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.cap = cv2.VideoCapture(file_path)

    def increment_leg_lift_count(self,angle1,count1,count2,count3):
        if (angle1 > 60):
            self.count1 = True
        if (self.count1 == True and angle1 > 100):
            self.count2 = True
        if (self.count2 == True and angle1 < 60):
            self.count3 = True
        if (self.count1 == True and self.count2 == True and self.count3 == True):
            self.leg_lift_count = self.leg_lift_count + 1
            self.count1 = False
            self.count2 = False
            self.count3 = False
        self.leg_lift_count_label.config(text="Leg Lift Count: {}".format(self.leg_lift_count))

    def start_video(self):
        self.video_running = True
        # Start a new thread to read and display video frames continuously
        threading.Thread(target=self.update_video).start()

    def stop_video(self):
        self.video_running = False

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image)

            annotated_image = image.copy()
            self.mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

            left_hip = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_HIP]

            midpoint = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)

            left_knee = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_KNEE]
            right_knee = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_KNEE]

            angle1 = angle_between_lines(left_knee.x, left_knee.y, midpoint[0], midpoint[1], right_knee.x, right_knee.y)

            self.increment_leg_lift_count(angle1,count1,count2,count3)

            self.angle_label.config(text="Angle: {:.2f}".format(angle1))

            cv2.circle(annotated_image, (int(midpoint[0] * annotated_image.shape[1]), int(midpoint[1] * annotated_image.shape[0])), 5, (255, 0, 0), -1)

            if angle1 >= 100 and angle1 <= 120:
                self.angle_label.config(fg="red", text="Angle: {:.2f}\nCorrect Exercise".format(angle1), font=("Helvetica", 26, "bold"))
                self.increment_leg_lift_count(angle1,count1,count2,count3)
            else:
                self.angle_label.config(fg="black", text="Angle: {:.2f}\nIncorrect Exercise".format(angle1), font=("Helvetica", 26, "bold"))

            # Resize the image to match the size of the canvas
            resized_image = cv2.resize(annotated_image, (640, 480))
            self.leg_lift_image = ImageTk.PhotoImage(Image.fromarray(resized_image))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.leg_lift_image)

        if self.video_running:
            self.master.after(10, self.update_video)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

def angle_between_lines(x1, y1, x2, y2, x3, y3):
    slope1 = (y2 - y1) / (x2 - x1)
    slope2 = (y3 - y2) / (x3 - x2)
    angle = math.atan2(slope2 - slope1, 1 + slope1 * slope2)
    return math.degrees(angle)

if __name__ == "__main__":
    root = tk.Tk()
    count1 = False
    count2 = False
    count3  = False
    app = LegLiftDetectorGUI(root, 0)
    root.configure(background='#b0c793')
    root.mainloop()
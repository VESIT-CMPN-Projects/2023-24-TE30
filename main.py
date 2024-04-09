import tkinter as tk
from tkinter import ttk, PhotoImage
from tkinter import *
import subprocess


def start_exercise():
    selected_exercise = exercise_combobox.get()
    if selected_exercise == "Shoulder Press":
        status_label.config(text=f"Please wait, Starting exercise: {selected_exercise}...")
        subprocess.Popen(["python", "exercise_models/shoulder_press.py"])
    elif selected_exercise == "Bicep Curl":
        status_label.config(text=f"Please wait, Starting exercise: {selected_exercise}...")
        subprocess.Popen(["python", "exercise_models/bicep_curl.py"])
    elif selected_exercise == "Plank":
        status_label.config(text=f"Please wait, Starting exercise: {selected_exercise}...")
        subprocess.Popen(["python", "exercise_models/Plank.py"])
    elif selected_exercise == "Leg Lifting":
        status_label.config(text=f"Please wait, Starting exercise: {selected_exercise}...")
        subprocess.Popen(["python", "exercise_models/side_lying_leg_lifting.py"])
    elif selected_exercise == "Tricep":
        status_label.config(text=f"Please wait, Starting exercise: {selected_exercise}...")
        subprocess.Popen(["python", "exercise_models/test.py"])
    elif selected_exercise == "Lunges":
        status_label.config(text=f"Please wait, Starting exercise: {selected_exercise}...")
        subprocess.Popen(["python", "exercise_models/lunges.py"])
    else:
        status_label.config(text="Please select an exercise.")


# Create the main window
root = tk.Tk()
root.title("Fitvision")
root.geometry("600x400")
root.resizable(False, False)  # Fix window size

bg: PhotoImage = PhotoImage(file="background_image2.png")
label1 = Label(root, image=bg)
label1.place(y=0, x=0)

# Create a label
title_label = ttk.Label(root, text="FitVision: Smart Posture Analysis for Effective Workouts", background="#b0c793",
                        font=("Helvetica", 16), foreground="#252518")
title_label.pack(pady=10)

instruction_label = ttk.Label(root, text="Select an exercise:", font=("Helvetica", 14), background="#b0c793")
instruction_label.place(x=150, y=120)
# instruction_label.pack()

# Create a combo box for exercise selection with a rounded button
exercises = ["Select an exercise", "Shoulder Press", "Bicep Curl", "Plank", "Leg Lifting", "Tricep"]
style = ttk.Style()
style.map("TCombobox", fieldbackground=[("readonly", "white")])
exercise_combobox = ttk.Combobox(root, values=exercises, state="readonly", font=("Helvetica", 12), background="#b0c793", width=20)
exercise_combobox.set(exercises[0])
exercise_combobox.place(x=320, y=120)
# exercise_combobox.pack(padx=5, pady=10)

# Create a button to start the exercise with rounded edges
icon_image = tk.PhotoImage(file="Running.png")
style.configure("TButton", borderwidth=0, relief="flat", font=("Helvetica", 11), background="#b0c793", color ="#b0c793")
start_button = ttk.Button(root, text="Start Exercise", command=start_exercise, image=icon_image, compound="left")
start_button.place(x=320, y=150)
# start_button.pack()

# Create a label for displaying the exercise status
status_label = ttk.Label(root, font=("Helvetica", 12), background="#b0c793")
status_label.place(x=250, y=220)
# status_label.pack(pady=10)

# Run the GUI
root.mainloop()

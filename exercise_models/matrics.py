import tkinter as tk
from tkinter import ttk, PhotoImage
from tkinter import *
import subprocess

# Create a function to update the metrics
def update_metrics():
    # You can replace these dummy values with actual values from your model
    accuracy = 75  # Example accuracy percentage
    exercise_count = 10  # Example exercise count
    angles = "45°, 90°, 120°"  # Example angles
    suggestions = "Keep your back straight"  # Example suggestions

    # Update the accuracy progress bar
    accuracy_bar["value"] = accuracy

    # Update the exercise count label
    count_label.config(text=f"Count: {exercise_count}")

    # Update the angles label
    angles_label.config(text=f"Angles: {angles}")

    # Update the suggestions label
    suggestions_label.config(text=f"Suggestions: {suggestions}")

    # Schedule the function to run again after a delay (e.g., 1000 milliseconds or 1 second)
    root.after(1000, update_metrics)

# Create the main window
root = tk.Tk()
root.title("Fitvision")
root.geometry("500x600")
root.resizable(False, False)  # Fix window size

bg: PhotoImage = PhotoImage(file="../matrics.png")
label1 = Label(root, image=bg)
label1.place(y=0, x=0)

# Create an accuracy progress bar
accuracy_bar_label = ttk.Label(root, text="Your Progress:", font=("Helvetica", 14), background="#b0c793")
accuracy_bar_label.place(x=70, y=150)
accuracy_bar = ttk.Progressbar(root, length=250, mode="determinate")
accuracy_bar.place(x=210, y=150.5)

# Create a label for exercise count
count_label = ttk.Label(root, font=("Helvetica", 14), background="#b0c793")
count_label.place(x=70, y=200)

# Create a label for displaying angles
angles_label = ttk.Label(root, font=("Helvetica", 14), background="#b0c793")
angles_label.place(x=70, y=250)

# Create a label for displaying suggestions
suggestions_label = ttk.Label(root, font=("Helvetica", 14), background="#b0c793")
suggestions_label.place(x=70, y=300)

# Start updating the metrics
update_metrics()

# Run the GUI
root.mainloop()

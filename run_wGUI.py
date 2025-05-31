import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from picamera2 import Picamera2
import time
from tensorflow.keras.models import load_model
import copy
from collections import deque
import tensorflow as tf
import itertools
import re
import tkinter as tk
from tkinter import Text
from PIL import Image, ImageTk
import threading
import subprocess

from gesture_classifier import GestureClassifier
gesture_classifier = GestureClassifier()

from output_handler import convert_input_to_output_best, score, substitute_placeholders

PIPER_MODEL = "en_US-lessac-medium.onnx"  # Ensure this file exists
OUTPUT_FILE = "output.wav"

# MediaPipe Holistic Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

actions = np.array(['hello', 'please', 'thanks', 'receipt', 'more', 'price', 'order', 'wait', 'bag', 'water', '0', '1', '2', '3', '4', '5', '(6W)', '7', '8', '(9F)', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z'])

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (480,640)}))
picam2.start()
time.sleep(2)  # Allow camera to stabilize

# Tkinter GUI Setup
root = tk.Tk()
root.attributes('-zoomed', True)
root.title("ASL Translator")
root.geometry("800x450")

latest_frame = None
output_buffer = []

def draw_landmarks(frame, results):
    # Draw multi hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                None,
                mp_drawing_styles.get_default_hand_landmarks_style(),
            )

def gather_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1],image.shape[0]

    landmark_point = []

    for _,landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark_x, landmark_y, landmark_z])

    return landmark_point

def pre_process_landmarks(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0,0

    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs,temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def extract_keypoints(frame, results):

    processed_hands = {"Right": None, "Left" : None}
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
    
            landmark_list = gather_landmark_list(frame, hand_landmarks)
    
            processed_landmarks = pre_process_landmarks(landmark_list)
    
            processed_hands[hand_label] = processed_landmarks

    if processed_hands["Right"] is None:
        processed_hands["Right"] = [0] * (21*3)
    if processed_hands["Left"] is None:
        processed_hands["Left"] = [0] * (21*3)
    
    return np.array(processed_hands["Right"] + processed_hands["Left"])

def text_to_speech(text):
    def run_tts():
        try:
            if not text.strip():  # Avoid speaking empty text
                print("No text to speak.")
                return

            root.after(0, status_label.config, {"text": "Playing Translation..."})  

            print("Generating speech...")
            
            # Run Piper command (Non-blocking)
            subprocess.run(
                ["piper", "--model", PIPER_MODEL, "--output_file", OUTPUT_FILE],
                input=text,
                text=True,
                check=True
            )

            print("Playing audio...")
            # Play the generated audio file (Non-blocking)
            subprocess.run(["aplay", OUTPUT_FILE], check=True)

        except Exception as e:
            print(f"Error: {e}")

        finally:
            root.after(0, status_label.config, {"text": ""})

    threading.Thread(target=run_tts, daemon=True).start()

# Function to update Picamera2 feed in the GUI
def update_frame():
    global latest_frame
    if latest_frame is not None:
        frame = cv2.resize(latest_frame, (480, 400))  # Resize for display
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
    root.after(10, update_frame)

# Function to clear the text box
def clear_text():
    global output_buffer
    output_buffer.clear()
    text_box.config(state="normal")  # Enable editing temporarily
    text_box.delete("1.0", tk.END)
    text_box.config(state="disabled")  # Disable editing again

# Function to update translation text
def update_translation(text):
    text_box.config(state="normal")  # Enable text update
    text_box.delete("1.0", tk.END)
    text_box.insert(tk.END, text)
    text_box.config(state="disabled")  # Disable editing after update

# Function to handle "Speak" button
def speak_text():
    global output_buffer
    if output_buffer:
        text_to_speech(" ".join(output_buffer))  # Convert buffer to speech

def translate_text():
    global output_buffer
    if output_buffer:
        new_buffer = convert_input_to_output_best(output_buffer)
        output_buffer = copy.deepcopy(new_buffer)

# OpenCV Feed Label (Top-Left)
lbl_video = tk.Label(root)
lbl_video.place(x=10, y=10, width=350, height=350)

# Non-Editable Translation Text Box (Center)
text_box = Text(root, wrap="word", font=("Arial", 18))
text_box.place(x=365, y=10, width=430, height=150)
text_box.config(state="disabled")  # Make the textbox non-editable

# Buttons
btn_clear = tk.Button(root, text="Translate", font=("Arial", 12), command=translate_text)
btn_clear.place(x=385, y=180, width=100, height=40)

btn_clear = tk.Button(root, text="Clear", font=("Arial", 12), command=clear_text)
btn_clear.place(x=495, y=180, width=100, height=40)

btn_speak = tk.Button(root, text="Speak", font=("Arial", 12), command=speak_text)
btn_speak.place(x=605, y=180, width=100, height=40)

# Labels
status_label = tk.Label(root, text="", font=("Arial", 12))
status_label.place(x=475, y=250, width=150, height=30)

instruction_label = tk.Label(root, text="please do signs slowly", font=("Arial", 12))
instruction_label.place(x=385, y=350, width=180, height=30)

def process_gestures():
    global latest_frame, output_buffer
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        sequence = deque(maxlen=15)
        confidence = 0.0
        debounce_buffer = deque(maxlen=5)  # Stores last few predictions
        repetition_buffer = deque(maxlen=20)
        CONSISTENCY_THRESHOLD = 3  # Required occurrences in buffer before accepting a sign
        CONFIDENCE_THRESHOLD = 0.90  # Minimum confidence to consider a prediction
    
        # Pre-fill sequence buffer with zero arrays to allow immediate model input
        dummy_keypoints = np.zeros((126,), dtype=np.float32)
        for _ in range(15):
            sequence.append(dummy_keypoints)
    
        while True:
            # Capture frame
            frame = picam2.capture_array()
            frame = cv2.flip(frame, -1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
            # Get frame dimensions
            height, width, _ = frame.shape
        
            # Apply zoom (center crop)
            zoom_factor = 1.5
            new_width = int(width / zoom_factor)
            new_height = int(height / zoom_factor)
            x_start = (width - new_width) // 2
            y_start = (height - new_height) // 2
            cropped_frame = frame[y_start:y_start+new_height, x_start:x_start+new_width]
        
            # Resize back to original frame dimensions
            zoomed_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_LINEAR)
    
            # Process the zoomed RGB frame with MediaPipe Hands
            results = hands.process(zoomed_frame)
            
            root.after(0, update_translation, " ".join(output_buffer))  # Instant UI update
            
            if results.multi_hand_landmarks:
                draw_landmarks(zoomed_frame, results)
                # Extract keypoints
                keypoints = extract_keypoints(zoomed_frame, results)
                sequence.append(keypoints)
    
                input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)  # Shape: (1, 15, 126)
                result_index, confidence = gesture_classifier(input_data)  # Use the TFLite model
    
                if confidence > CONFIDENCE_THRESHOLD:
                    new_prediction = actions[result_index]
                    
                    debounce_buffer.append(new_prediction)
                    
                    # Accept new_prediction only if it appears multiple times in temp buffer
                    if debounce_buffer.count(new_prediction) >= CONSISTENCY_THRESHOLD:
                        if not output_buffer or output_buffer[-1] != new_prediction:
                            output_buffer.append(new_prediction)

                        repetition_buffer.append(new_prediction)

                        if len(repetition_buffer) == repetition_buffer.maxlen and all(p == new_prediction for p in repetition_buffer):
                            output_buffer.append(new_prediction)
                            repetition_buffer.clear()
                            
                        root.after(0, update_translation, " ".join(output_buffer))
            else:
                sequence.append(dummy_keypoints)  # Add zeros instead of clearing
                debounce_buffer.clear()

            latest_frame = zoomed_frame

threading.Thread(target=process_gestures, daemon=True).start()

update_frame()

root.mainloop()
picam2.stop()
picam2.close()
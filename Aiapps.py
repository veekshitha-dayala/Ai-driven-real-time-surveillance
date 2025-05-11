import cv2
import time
from datetime import datetime
from PIL import Image
import customtkinter as ctk
from customtkinter import CTkImage
import os

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Surveillance App")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Default settings
        self.min_record_time = 10
        self.stop_delay = 10
        self.face_cascade = cv2.CascadeClassifier(r"C:\Users\HP\OneDrive\Desktop\Ai app\haarcascade_frontalface_default.xml")

        self.cap = None
        self.running = False
        self.recording = False
        self.out = None
        self.output_file = None
        self.record_start_time = None
        self.last_detect_time = None
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Initialize the GUI components
        self.video_label = ctk.CTkLabel(self.root)
        self.video_label.pack(padx=10, pady=10)

        self.init_settings_panel()

    def init_settings_panel(self):
        frame = ctk.CTkFrame(self.root)
        frame.pack(padx=10, pady=10)

        self.record_time_label = ctk.CTkLabel(frame, text="Min Record Time (seconds):")
        self.record_time_label.pack()
        self.record_time_entry = ctk.CTkEntry(frame)
        self.record_time_entry.insert(0, str(self.min_record_time))
        self.record_time_entry.pack()

        self.stop_delay_label = ctk.CTkLabel(frame, text="Stop Delay (seconds):")
        self.stop_delay_label.pack()
        self.stop_delay_entry = ctk.CTkEntry(frame)
        self.stop_delay_entry.insert(0, str(self.stop_delay))
        self.stop_delay_entry.pack()

        self.update_button = ctk.CTkButton(frame, text="Update Settings", command=self.update_settings)
        self.update_button.pack(pady=10)

        self.start_button = ctk.CTkButton(frame, text="Start Webcam", command=self.start_webcam)
        self.start_button.pack(pady=5)

        self.stop_button = ctk.CTkButton(frame, text="Stop Webcam", command=self.stop_webcam)
        self.stop_button.pack(pady=5)

    def update_settings(self):
        try:
            self.min_record_time = int(self.record_time_entry.get())
            self.stop_delay = int(self.stop_delay_entry.get())
            print(f"Updated: Min Record Time = {self.min_record_time}, Stop Delay = {self.stop_delay}")
        except ValueError:
            print("Invalid input. Please enter valid integer values.")

    def start_webcam(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam.")
                return
            self.running = True
            self.update_frame()

    def stop_webcam(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.recording and self.out:
            self.out.release()
        self.video_label.configure(image=None)
        self.video_label.image = None

    def update_frame(self):
        if self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            current_time = time.time()

            # Draw a small circle to indicate recording status
            circle_color = (0, 255, 0) if self.recording else (0, 0, 255)
            cv2.circle(frame, (20, 20), 10, circle_color, -1)

            # Start or continue recording if faces are detected
            if len(faces) > 0:
                if not self.recording:
                    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    folder = "normal"
                    os.makedirs(folder, exist_ok=True)
                    self.output_file = f"{folder}/output_{timestamp}.avi"
                    height, width, _ = frame.shape
                    self.out = cv2.VideoWriter(self.output_file, self.fourcc, 20.0, (width, height))
                    self.record_start_time = current_time
                    print(f"Recording started: {self.output_file}")
                self.recording = True
                self.last_detect_time = current_time
                self.log_event(faces=len(faces))
            else:
                if self.recording and self.last_detect_time and (current_time - self.last_detect_time) > self.stop_delay:
                    duration = current_time - self.record_start_time
                    self.out.release()
                    if duration < self.min_record_time:
                        os.remove(self.output_file)
                        print("Recording discarded due to short duration.")
                    else:
                        print(f"Recording saved: {self.output_file}")
                    self.recording = False
                    self.out = None

            if self.recording and self.out:
                self.out.write(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img_tk = CTkImage(light_image=img, size=(frame.shape[1], frame.shape[0]))
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

            self.root.after(10, self.update_frame)

    def log_event(self, faces):
        with open("logs.txt", "a") as log:
            log.write(f"{datetime.now()} | File: {self.output_file} | Faces: {faces}\n")

if __name__ == "__main__":
    root = ctk.CTk()
    app = FaceDetectionApp(root)
    root.mainloop()

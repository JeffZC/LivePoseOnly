import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import csv
from datetime import datetime
import time

class PoseTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Pose Tracker - No Video Recording")
        self.root.geometry("1000x700")
        
        # MediaPipe setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = None
        
        # Camera and display settings
        self.cap = None
        self.camera_index = 1
        self.available_cameras = []
        self.is_running = False
        self.display_mode = 2  # 0=normal, 1=blurred, 2=pose only (default: pose only)
        self.mode_names = ["Normal Video", "Blurred Video", "Pose Only"]
        
        # CSV recording
        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None
        self.frame_count = 0
        
        # FPS tracking
        self.fps_start_time = 0
        self.fps_counter = 0
        
        # Canvas image object
        self.canvas_image = None
        
        self.setup_ui()
        self.detect_cameras()
        
    def setup_ui(self):
        # Top control panel
        control_frame = tk.Frame(self.root, bg="#2c3e50", height=120)
        control_frame.pack(fill=tk.X, side=tk.TOP)
        control_frame.pack_propagate(False)
        
        # Camera selection
        tk.Label(control_frame, text="Camera:", bg="#2c3e50", fg="white", font=("Arial", 10)).place(x=10, y=10)
        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(control_frame, textvariable=self.camera_var, state="readonly", width=15)
        self.camera_dropdown.place(x=10, y=35)
        self.camera_dropdown.bind("<<ComboboxSelected>>", self.change_camera)
        
        # Start/Stop button
        self.start_btn = tk.Button(control_frame, text="▶ Start Camera", command=self.toggle_camera, 
                                    bg="#27ae60", fg="white", font=("Arial", 12, "bold"), width=15)
        self.start_btn.place(x=10, y=70)
        
        # Display mode buttons
        tk.Label(control_frame, text="Display Mode:", bg="#2c3e50", fg="white", font=("Arial", 10)).place(x=200, y=10)
        self.mode_btn = tk.Button(control_frame, text="🔄 Toggle Mode", command=self.toggle_display_mode,
                                   bg="#3498db", fg="white", font=("Arial", 10), width=15)
        self.mode_btn.place(x=200, y=35)
        self.mode_label = tk.Label(control_frame, text="Mode: Pose Only", bg="#2c3e50", fg="#ecf0f1", font=("Arial", 9))
        self.mode_label.place(x=200, y=70)
        
        # Recording controls
        tk.Label(control_frame, text="CSV Recording:", bg="#2c3e50", fg="white", font=("Arial", 10)).place(x=400, y=10)
        self.record_btn = tk.Button(control_frame, text="⏺ Start Recording", command=self.toggle_recording,
                                     bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), width=15, state=tk.DISABLED)
        self.record_btn.place(x=400, y=35)
        self.record_status = tk.Label(control_frame, text="Not Recording", bg="#2c3e50", fg="#95a5a6", font=("Arial", 9))
        self.record_status.place(x=400, y=70)
        
        # FPS and frame counter
        self.fps_label = tk.Label(control_frame, text="FPS: 0", bg="#2c3e50", fg="#ecf0f1", font=("Arial", 10))
        self.fps_label.place(x=600, y=10)
        self.frame_label = tk.Label(control_frame, text="Frames: 0", bg="#2c3e50", fg="#ecf0f1", font=("Arial", 10))
        self.frame_label.place(x=600, y=35)
        
        # Warning label
        warning = tk.Label(control_frame, text="⚠ VIDEO IS NOT SAVED - Only pose data can be exported to CSV", 
                          bg="#2c3e50", fg="#f39c12", font=("Arial", 9, "italic"))
        warning.place(x=200, y=95)
        
        # Video display canvas
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg="#34495e", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(status_frame, text="Ready - Select camera and click Start", 
                                     bg="#34495e", fg="white", font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
    def detect_cameras(self):
        """Detect available cameras"""
        self.available_cameras = []
        for i in range(5):
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                self.available_cameras.append(i)
                temp_cap.release()
        
        if self.available_cameras:
            self.camera_dropdown['values'] = [f"Camera {i}" for i in self.available_cameras]
            default_cam = 1 if 1 in self.available_cameras else self.available_cameras[0]
            self.camera_var.set(f"Camera {default_cam}")
            self.camera_index = default_cam
            self.status_label.config(text=f"Found {len(self.available_cameras)} camera(s)")
        else:
            messagebox.showerror("Error", "No cameras detected!")
            self.status_label.config(text="No cameras found")
            
    def change_camera(self, event=None):
        """Change camera selection"""
        if self.is_running:
            messagebox.showwarning("Warning", "Stop camera before changing selection")
            return
        
        selected = self.camera_var.get()
        self.camera_index = int(selected.split()[1])
        self.status_label.config(text=f"Selected Camera {self.camera_index}")
        
    def toggle_camera(self):
        """Start or stop camera"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open camera {self.camera_index}")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.is_running = True
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.start_btn.config(text="⏹ Stop Camera", bg="#e74c3c")
        self.record_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Camera {self.camera_index} running")
        
        # Start video loop with after()
        self.update_frame()
        
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        
        if self.is_recording:
            self.stop_recording()
        
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()
            
        self.start_btn.config(text="▶ Start Camera", bg="#27ae60")
        self.record_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Camera stopped")
        self.canvas.delete("all")
        
    def toggle_display_mode(self):
        """Toggle display mode"""
        self.display_mode = (self.display_mode + 1) % 3
        self.mode_label.config(text=f"Mode: {self.mode_names[self.display_mode]}")
        
    def toggle_recording(self):
        """Start or stop CSV recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Start CSV recording"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"pose_data_{timestamp}.csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not filename:
            return
            
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        header = ['timestamp', 'frame']
        for i in range(33):  # MediaPipe has 33 landmarks
            header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z', f'landmark_{i}_visibility'])
        self.csv_writer.writerow(header)
        
        self.is_recording = True
        self.record_btn.config(text="⏹ Stop Recording", bg="#27ae60")
        self.record_status.config(text="🔴 Recording...", fg="#e74c3c")
        self.status_label.config(text=f"Recording to: {filename}")
        
    def stop_recording(self):
        """Stop CSV recording"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            
        self.is_recording = False
        self.record_btn.config(text="⏺ Start Recording", bg="#e74c3c")
        self.record_status.config(text="Not Recording", fg="#95a5a6")
        self.status_label.config(text="Recording stopped")
        
    def update_frame(self):
        """Update video frame"""
        if not self.is_running:
            return
            
        success, frame = self.cap.read()
        if not success:
            self.root.after(10, self.update_frame)
            return
            
        self.frame_count += 1
        self.fps_counter += 1
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        # Prepare display frame based on mode
        if self.display_mode == 0:  # Normal
            display_frame = frame.copy()
        elif self.display_mode == 1:  # Blurred
            display_frame = cv2.GaussianBlur(frame, (51, 51), 0)
        else:  # Pose only
            display_frame = np.zeros_like(frame)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                display_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Save to CSV if recording
            if self.is_recording and self.csv_writer:
                row = [datetime.now().isoformat(), self.frame_count]
                for landmark in results.pose_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                self.csv_writer.writerow(row)
        
        # Convert to PhotoImage for Tkinter
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_frame)
        img = img.resize((800, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=img)
        
        # Update canvas more efficiently
        if self.canvas_image is None:
            self.canvas_image = self.canvas.create_image(400, 300, image=photo)
        else:
            self.canvas.itemconfig(self.canvas_image, image=photo)
        self.canvas.image = photo  # Keep reference
        
        # Update FPS
        if time.time() - self.fps_start_time >= 1.0:
            self.fps_label.config(text=f"FPS: {self.fps_counter}")
            self.fps_counter = 0
            self.fps_start_time = time.time()
            
        self.frame_label.config(text=f"Frames: {self.frame_count}")
        
        # Schedule next frame update (30ms = ~33 FPS)
        self.root.after(10, self.update_frame)
            
    def on_closing(self):
        """Handle window close"""
        if self.is_running:
            self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseTrackerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
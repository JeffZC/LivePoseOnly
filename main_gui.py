import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import csv
from datetime import datetime
import time
import sys  # Add this at the top with other imports
import os  # Add this at the top with other imports

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
        self.blur_alpha = 0.5  # Transparency for blurred mode (0=opaque, 1=transparent)
        
        # Camera resolution (will be set when camera opens)
        self.camera_width = 640
        self.camera_height = 480
        self.camera_aspect_ratio = 4 / 3
        
        # CSV recording
        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None
        self.frame_count = 0
        self.export_format = "MediaPipe 33"  # Default format
        self.csv_buffer = []  # Buffer for CSV rows
        
        # Landmark configurations
        self.landmarks_config = None
        self.include_visibility = False
        
        # FPS tracking
        self.fps_start_time = 0
        self.fps_counter = 0
        
        # Canvas image object
        self.canvas_image = None
        
        # Cached display size
        self.display_width = 800
        self.display_height = 600
        self.last_canvas_width = 0
        self.last_canvas_height = 0
        
        # Track after() callback ID
        self.after_id = None
        
        self.setup_ui()
        self.detect_cameras()
        
        # Bind canvas resize event
        self.canvas.bind('<Configure>', self.on_canvas_resize)
    
    def setup_ui(self):
        # Top control panel
        control_frame = tk.Frame(self.root, bg="#2c3e50", height=140)
        control_frame.pack(fill=tk.X, side=tk.TOP)
        control_frame.pack_propagate(False)
        
        # Camera selection
        tk.Label(control_frame, text="Camera:", bg="#2c3e50", fg="white").place(x=10, y=10)
        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(control_frame, textvariable=self.camera_var, state="readonly", width=18)
        self.camera_dropdown.place(x=10, y=35)
        self.camera_dropdown.bind("<<ComboboxSelected>>", self.change_camera)
        
        # Start/Stop button
        self.start_btn = tk.Button(control_frame, text="▶ Start Camera", command=self.toggle_camera, 
                                    bg="#27ae60", fg="white", width=18)
        self.start_btn.place(x=10, y=70)
        
        # Display mode selection
        tk.Label(control_frame, text="Display Mode:", bg="#2c3e50", fg="white").place(x=250, y=10)
        self.mode_var = tk.StringVar(value="Pose Only")
        self.mode_dropdown = ttk.Combobox(control_frame, textvariable=self.mode_var, state="readonly", width=18)
        self.mode_dropdown['values'] = ["Normal Video", "Blurred Video", "Pose Only"]
        self.mode_dropdown.place(x=250, y=35)
        self.mode_dropdown.bind("<<ComboboxSelected>>", self.change_display_mode)
        
        # Blur transparency slider (with built-in label) - same width as dropdowns/buttons
        self.blur_slider = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                      command=self.update_blur_alpha, length=130,
                                      bg="#2c3e50", fg="white", highlightthickness=0,
                                      troughcolor="#34495e", activebackground="#3498db",
                                      label="Transparency", showvalue=True)
        self.blur_slider.set(50)  # Default 50% transparency
        
        # Hide initially
        self.blur_slider.place_forget()
        
        # Export format selection
        tk.Label(control_frame, text="Export Format:", bg="#2c3e50", fg="white").place(x=490, y=10)
        self.format_var = tk.StringVar(value="MediaPipe 33")
        self.format_dropdown = ttk.Combobox(control_frame, textvariable=self.format_var, state="readonly", width=18)
        self.format_dropdown['values'] = ["MediaPipe 33", "RR21"]
        self.format_dropdown.place(x=490, y=35)
        self.format_dropdown.bind("<<ComboboxSelected>>", self.change_format)
        
        # Recording controls
        self.record_btn = tk.Button(control_frame, text="⏺ Start Recording", command=self.toggle_recording,
                                     bg="#e74c3c", fg="white", width=18, state=tk.DISABLED)
        self.record_btn.place(x=490, y=70)
        self.record_status = tk.Label(control_frame, text="Not Recording", bg="#2c3e50", fg="#95a5a6")
        self.record_status.place(x=490, y=105)
        
        # FPS counter
        self.fps_label = tk.Label(control_frame, text="FPS: 0", bg="#2c3e50", fg="#ecf0f1")
        self.fps_label.place(x=730, y=10)
        
        # Video display canvas
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg="#34495e", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(status_frame, text="Ready - Select camera and click Start", 
                                     bg="#34495e", fg="white")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Warning label in status bar
        warning = tk.Label(status_frame, text="⚠ VIDEO NOT SAVED • Pose data only", 
                          bg="#34495e", fg="#f39c12")
        warning.pack(side=tk.RIGHT, padx=10)
        
    def detect_cameras(self):
        """Detect available cameras"""
        self.available_cameras = []
        for i in range(3):  # Check only first 3 cameras for faster startup
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
    
    def change_format(self, event=None):
        """Change export format"""
        self.export_format = self.format_var.get()
        self.status_label.config(text=f"Export format: {self.export_format}")
        
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
        
        # Optimize camera settings for speed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Read actual resolution the camera is using
        self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.camera_aspect_ratio = self.camera_width / self.camera_height
        
        print(f"Camera resolution: {self.camera_width}x{self.camera_height} (aspect ratio: {self.camera_aspect_ratio:.2f})")
        
        # Use Lite model for faster loading and processing
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
        self.status_label.config(text=f"Camera {self.camera_index}: {self.camera_width}x{self.camera_height} running")
        
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
        self.canvas_image = None  # Reset canvas image object
        self.frame_count = 0
        self.fps_counter = 0
        
    def on_canvas_resize(self, event):
        """Called only when canvas is resized"""
        if event.width != self.last_canvas_width or event.height != self.last_canvas_height:
            self.last_canvas_width = event.width
            self.last_canvas_height = event.height
            
            aspect_ratio = self.camera_aspect_ratio
            
            if event.width / event.height > aspect_ratio:
                self.display_height = event.height - 20
                self.display_width = int(self.display_height * aspect_ratio)
            else:
                self.display_width = event.width - 20
                self.display_height = int(self.display_width / aspect_ratio)
            
            self.display_width = max(320, self.display_width)
            self.display_height = max(240, self.display_height)
    
    def change_display_mode(self, event=None):
        """Change display mode via dropdown"""
        selected = self.mode_var.get()
        self.display_mode = self.mode_names.index(selected)
        
        # Show/hide blur slider based on mode
        if selected == "Blurred Video":
            self.blur_slider.place(x=250, y=70)
        else:
            self.blur_slider.place_forget()
    
    def update_blur_alpha(self, value):
        """Update blur transparency level"""
        self.blur_alpha = float(value) / 100.0
        
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
        
        # Define landmark configurations
        if self.export_format == "MediaPipe 33":
            # MediaPipe 33
            self.landmarks_config = [
                (0, 'nose'), (1, 'left_eye_inner'), (2, 'left_eye'), (3, 'left_eye_outer'),
                (4, 'right_eye_inner'), (5, 'right_eye'), (6, 'right_eye_outer'),
                (7, 'left_ear'), (8, 'right_ear'), (9, 'mouth_left'), (10, 'mouth_right'),
                (11, 'left_shoulder'), (12, 'right_shoulder'), (13, 'left_elbow'), (14, 'right_elbow'),
                (15, 'left_wrist'), (16, 'right_wrist'), (17, 'left_pinky'), (18, 'right_pinky'),
                (19, 'left_index'), (20, 'right_index'), (21, 'left_thumb'), (22, 'right_thumb'),
                (23, 'left_hip'), (24, 'right_hip'), (25, 'left_knee'), (26, 'right_knee'),
                (27, 'left_ankle'), (28, 'right_ankle'), (29, 'left_heel'), (30, 'right_heel'),
                (31, 'left_foot_index'), (32, 'right_foot_index')
            ]
        else:
            # RR21
            self.landmarks_config = [
                (0, 'nose'),
                (2, 'left_eye'), (5, 'right_eye'),
                (7, 'left_ear'), (8, 'right_ear'),
                (11, 'left_shoulder'), (12, 'right_shoulder'),
                (13, 'left_elbow'), (14, 'right_elbow'),
                (15, 'left_wrist'), (16, 'right_wrist'),
                (23, 'left_hip'), (24, 'right_hip'),
                (25, 'left_knee'), (26, 'right_knee'),
                (27, 'left_ankle'), (28, 'right_ankle'),
                (29, 'left_heel'), (30, 'right_heel'),
                (31, 'left_foot_index'), (32, 'right_foot_index')
            ]
        
        # Build header - consistent format for both: x, y, depth, visibility
        header = ['frame', 'timestamp']
        for idx, name in self.landmarks_config:
            header.extend([f'{name}_x', f'{name}_y', f'{name}_depth', f'{name}_visibility'])
        self.csv_writer.writerow(header)
        
        self.frame_count = 0
        self.is_recording = True
        self.record_btn.config(text="⏹ Stop Recording", bg="#27ae60")
        self.record_status.config(text="🔴 Recording... Frame: 0", fg="#e74c3c")
        self.status_label.config(text=f"Recording to: {filename} ({self.export_format})")
        
    def stop_recording(self):
        """Stop CSV recording"""
        if self.csv_file:
            # Write any remaining buffered rows
            if self.csv_buffer:
                self.csv_writer.writerows(self.csv_buffer)
                self.csv_buffer = []
            
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
            self.after_id = None  # Clear the callback ID
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
        elif self.display_mode == 1:  # Blurred with transparency
            blurred = cv2.GaussianBlur(frame, (51, 51), 0)
            black_bg = np.zeros_like(frame)
            # Blend: more alpha = more transparent (towards black)
            display_frame = cv2.addWeighted(blurred, 1 - self.blur_alpha, black_bg, self.blur_alpha, 0)
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
                landmarks = results.pose_landmarks.landmark
                timestamp = datetime.now().isoformat()
                
                # Build row - consistent format: x, y, depth, visibility
                row = [self.frame_count, timestamp]
                for idx, name in self.landmarks_config:
                    lm = landmarks[idx]
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                self.csv_buffer.append(row)
                
                # Write buffer every 60 frames
                if len(self.csv_buffer) >= 60:
                    self.csv_writer.writerows(self.csv_buffer)
                    self.csv_buffer = []
                    self.csv_file.flush()
        
        # Convert to PhotoImage for Tkinter
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_frame)
        
        # Resize with faster algorithm
        if (self.display_width != img.width or self.display_height != img.height):
            img = img.resize((self.display_width, self.display_height), Image.Resampling.BILINEAR)
        
        photo = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        center_x = self.last_canvas_width // 2
        center_y = self.last_canvas_height // 2
        
        if self.canvas_image is None:
            self.canvas_image = self.canvas.create_image(center_x, center_y, image=photo)
        else:
            self.canvas.coords(self.canvas_image, center_x, center_y)
            self.canvas.itemconfig(self.canvas_image, image=photo)
        self.canvas.image = photo
        
        # Update FPS
        if time.time() - self.fps_start_time >= 1.0:
            self.fps_label.config(text=f"FPS: {self.fps_counter}")
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        if self.is_recording:
            self.record_status.config(text=f"🔴 Recording... Frame: {self.frame_count}")
        
        # Schedule next update and save the ID
        self.after_id = self.root.after(10, self.update_frame)
            
    def on_closing(self):
        """Handle window close - standard Python/Tkinter cleanup"""
        # 1. Stop the update loop
        self.is_running = False
        
        # 2. Cancel pending callbacks
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        
        # 3. Close recording
        if self.is_recording:
            self.stop_recording()
        
        # 4. Release resources
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.pose is not None:
            self.pose.close()
            self.pose = None
        
        # 5. Standard Tkinter cleanup
        self.root.destroy()  # This is enough - no quit() needed

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseTrackerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
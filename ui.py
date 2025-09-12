import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np

class GameDetectorUI:
    def __init__(self, detector):
        self.detector = detector
        self.root = tk.Tk()
        self.root.title("Project IGI Game - Aimbot Control")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Detection state
        self.is_detection_running = False
        
        # UI Variables
        self.aim_status = tk.BooleanVar(value=False)
        self.target_lock_status = tk.BooleanVar(value=False)
        
        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Project IGI Game - Aimbot Control", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Control Panel Frame
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        control_frame.columnconfigure(1, weight=1)
        
        # Start/Stop Detection Button
        self.start_stop_btn = ttk.Button(control_frame, text="Start Detection", 
                                        command=self.toggle_detection)
        self.start_stop_btn.grid(row=0, column=0, padx=(0, 10), pady=5)
        
        # Status indicator
        self.status_label = ttk.Label(control_frame, text="Status: Stopped", 
                                     foreground="red")
        self.status_label.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Aim Control Frame
        aim_frame = ttk.LabelFrame(main_frame, text="Aim Controls", padding="10")
        aim_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        aim_frame.columnconfigure(1, weight=1)
        
        # Aim Toggle Button
        self.aim_btn = ttk.Button(aim_frame, text="Toggle Aim", 
                                 command=self.toggle_aim)
        self.aim_btn.grid(row=0, column=0, padx=(0, 10), pady=5)
        
        # Aim Status
        self.aim_status_label = ttk.Label(aim_frame, text="Aim: OFF", 
                                         foreground="red")
        self.aim_status_label.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Target Lock Toggle Button
        self.target_lock_btn = ttk.Button(aim_frame, text="Toggle Target Lock", 
                                         command=self.toggle_target_lock)
        self.target_lock_btn.grid(row=1, column=0, padx=(0, 10), pady=5)
        
        # Target Lock Status
        self.target_lock_status_label = ttk.Label(aim_frame, text="Target Lock: OFF", 
                                                 foreground="red")
        self.target_lock_status_label.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Settings Frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        settings_frame.columnconfigure(1, weight=1)
        
        # Detection Threshold
        ttk.Label(settings_frame, text="Detection Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.threshold_var = tk.DoubleVar(value=0.6)
        threshold_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
        
        # Aim Smoothing
        ttk.Label(settings_frame, text="Aim Smoothing:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.smoothing_var = tk.DoubleVar(value=0.2)
        smoothing_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                   variable=self.smoothing_var, orient=tk.HORIZONTAL)
        smoothing_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
        
        # Mouse Sensitivity
        ttk.Label(settings_frame, text="Mouse Sensitivity:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.sensitivity_var = tk.DoubleVar(value=0.7)
        sensitivity_scale = ttk.Scale(settings_frame, from_=0.1, to=2.0, 
                                     variable=self.sensitivity_var, orient=tk.HORIZONTAL)
        sensitivity_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
        
        # Detection Info Frame
        info_frame = ttk.LabelFrame(main_frame, text="Detection Info", padding="10")
        info_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        
        # Detection Status
        ttk.Label(info_frame, text="Detection Status:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.detection_info_label = ttk.Label(info_frame, text="No targets detected", 
                                            foreground="gray")
        self.detection_info_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Target Count
        ttk.Label(info_frame, text="Targets Found:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.target_count_label = ttk.Label(info_frame, text="0", foreground="blue")
        self.target_count_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Stats Frame
        stats_frame = ttk.Frame(main_frame)
        stats_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))
        stats_frame.columnconfigure(1, weight=1)
        
        # FPS Label
        self.fps_label = ttk.Label(stats_frame, text="FPS: 0")
        self.fps_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        # Shot Count Label
        self.shot_count_label = ttk.Label(stats_frame, text="Shots: 0")
        self.shot_count_label.grid(row=0, column=1, sticky=tk.W)
        
    def setup_bindings(self):
        # Bind settings changes
        self.threshold_var.trace('w', self.update_settings)
        self.smoothing_var.trace('w', self.update_settings)
        self.sensitivity_var.trace('w', self.update_settings)
        
        # Bind window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def update_settings(self, *args):
        """Update detector settings from UI controls"""
        self.detector.detection_threshold = self.threshold_var.get()
        self.detector.aim_smoothing = self.smoothing_var.get()
        self.detector.mouse_sensitivity = self.sensitivity_var.get()
        
    def toggle_detection(self):
        """Toggle detection on/off"""
        if not self.is_detection_running:
            self.start_detection()
        else:
            self.stop_detection()
            
    def start_detection(self):
        """Start detection using timer-based approach"""
        try:
            self.is_detection_running = True
            self.start_stop_btn.config(text="Stop Detection")
            self.status_label.config(text="Status: Running", foreground="green")
            
            # Initialize detector
            self.detector.start_detection()
            
            # Start timer-based detection loop
            self.detection_timer()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
            self.is_detection_running = False
            self.start_stop_btn.config(text="Start Detection")
            self.status_label.config(text="Status: Error", foreground="red")
            
    def stop_detection(self):
        """Stop detection"""
        self.is_detection_running = False
        self.detector.stop_detection()
        self.start_stop_btn.config(text="Start Detection")
        self.status_label.config(text="Status: Stopped", foreground="red")
        
    def detection_timer(self):
        """Timer-based detection loop"""
        if not self.is_detection_running:
            return
            
        try:
            start_time = cv2.getTickCount()
            
            # Capture and process frame
            frame = self.detector.capture_screen()
            tracked_detections = self.detector.process_frame(frame)
            
            # Calculate FPS
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - start_time)
            
            # Update UI
            self.update_ui(tracked_detections, fps)
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            messagebox.showerror("Detection Error", str(e))
            self.stop_detection()
            return
        
        # Schedule next detection cycle (approximately 60 FPS for better performance)
        self.root.after(16, self.detection_timer)
            
    def update_ui(self, tracked_detections, fps):
        """Update UI elements with detection stats"""
        try:
            # Update stats
            self.fps_label.config(text=f"FPS: {int(fps)}")
            self.shot_count_label.config(text=f"Shots: {self.detector.shot_count}")
            
            # Update detection info
            if len(tracked_detections) > 0:
                self.detection_info_label.config(text="Targets detected", foreground="green")
                self.target_count_label.config(text=str(len(tracked_detections)), foreground="green")
            else:
                self.detection_info_label.config(text="No targets detected", foreground="gray")
                self.target_count_label.config(text="0", foreground="gray")
            
        except Exception as e:
            print(f"UI update error: {str(e)}")
            
    def toggle_aim(self):
        """Toggle aim on/off"""
        aim_enabled = self.detector.toggle_aim()
        self.aim_status.set(aim_enabled)
        status_text = "Aim: ON" if aim_enabled else "Aim: OFF"
        color = "green" if aim_enabled else "red"
        self.aim_status_label.config(text=status_text, foreground=color)
        
    def toggle_target_lock(self):
        """Toggle target lock on/off"""
        target_lock_enabled = self.detector.toggle_target_lock()
        self.target_lock_status.set(target_lock_enabled)
        status_text = "Target Lock: ON" if target_lock_enabled else "Target Lock: OFF"
        color = "green" if target_lock_enabled else "red"
        self.target_lock_status_label.config(text=status_text, foreground=color)
        
    def on_closing(self):
        """Handle window closing"""
        if self.is_detection_running:
            self.stop_detection()
        self.detector.cleanup()
        self.root.destroy()
        
    def run(self):
        """Start the UI main loop"""
        self.root.mainloop()

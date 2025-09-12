import numpy as np
import cv2
import mss
import torch
from time import time
import time
from ultralytics import YOLO
import win32api
import win32con
from math import sqrt
import supervision as sv

class GameDetector:
    def __init__(self, model_path):
        # Model initialization
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Screen capture setup - will be initialized when needed
        self.sct = None
        self.monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        
        # ByteTrack initialization with proper components
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        # Tracking settings
        self.detection_threshold = 0.6
        self.tracked_objects = []
        
        # Aim settings
        self.aim_enabled = False
        self.target_lock = False
        self.screen_center = (self.monitor['width'] // 2, self.monitor['height'] // 2)
        self.aim_threshold = 40
        self.shot_cooldown = 0.1
        self.last_shot_time = 0
        self.shot_count = 0
        self.aim_smoothing = 0.2
        self.last_aim_pos = self.screen_center
        self.mouse_sensitivity = 0.7
        
        # Detection state
        self.is_running = False

    def process_frame(self, frame):
        # Run YOLO detection
        results = self.model(frame, device=self.device)[0]
        
        # Convert YOLO results to supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter detections based on confidence
        mask = detections.confidence >= self.detection_threshold
        detections = detections[mask]
        
        # Update tracks using ByteTrack
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Process tracked objects for targeting
        best_target = None
        min_distance = float('inf')
        
        if len(tracked_detections) > 0:
            # Process each detection for targeting
            for i in range(len(tracked_detections)):
                xyxy = tracked_detections.xyxy[i]
                class_id = tracked_detections.class_id[i]
                class_name = self.model.names[class_id]
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Calculate distance for enemy class only
                if class_name.lower() == 'enemy':
                    target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    distance = self.calculate_distance(target_center, self.screen_center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_target = (x1, y1, x2, y2, class_name)
        
        # Handle aiming if we have a valid target
        if best_target and self.aim_enabled:
            self.aim_at_target(best_target[:4], best_target[4])
            
        return tracked_detections

    def calculate_distance(self, point1, point2):
        return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def move_mouse_relative(self, dx, dy):
        scaled_dx = int(dx * self.mouse_sensitivity)
        scaled_dy = int(dy * self.mouse_sensitivity)
        max_move = 50
        scaled_dx = max(min(scaled_dx, max_move), -max_move)
        scaled_dy = max(min(scaled_dy, max_move), -max_move)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, scaled_dx, scaled_dy, 0, 0)
    
    def shoot(self):
        current_time = time.time()
        if current_time - self.last_shot_time >= self.shot_cooldown:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.02)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self.last_shot_time = current_time
            self.shot_count += 1
            return True
        return False
    
    def aim_at_target(self, target_pos, target_class):
        if not self.aim_enabled or target_class.lower() != 'enemy':
            return False
            
        target_x = (target_pos[0] + target_pos[2]) // 2
        target_y = (target_pos[1] + target_pos[3]) // 2
        
        dx = target_x - self.screen_center[0]
        dy = target_y - self.screen_center[1]
        
        smooth_dx = int(dx * self.aim_smoothing)
        smooth_dy = int(dy * self.aim_smoothing)
        
        if abs(dx) > 5 or abs(dy) > 5:
            self.move_mouse_relative(smooth_dx, smooth_dy)
        
        distance = sqrt(dx*dx + dy*dy)
        if distance <= self.aim_threshold and self.target_lock:
            return self.shoot()
        
        return False

    def capture_screen(self):
        # Initialize mss if not already done
        if self.sct is None:
            self.sct = mss.mss()
        
        screenshot = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    
    def start_detection(self):
        self.is_running = True
        print("Detection started")
    
    def stop_detection(self):
        self.is_running = False
        print("Detection stopped")
    
    def toggle_aim(self):
        self.aim_enabled = not self.aim_enabled
        print(f"Aim {'enabled' if self.aim_enabled else 'disabled'}")
        return self.aim_enabled
    
    def toggle_target_lock(self):
        self.target_lock = not self.target_lock
        print(f"Target lock {'enabled' if self.target_lock else 'disabled'}")
        return self.target_lock
    
    def cleanup(self):
        if hasattr(self, 'sct'):
            self.sct.close()
        cv2.destroyAllWindows()

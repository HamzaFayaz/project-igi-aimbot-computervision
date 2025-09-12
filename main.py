#!/usr/bin/env python3
"""
Game Detector - Aimbot Application
Main entry point for the game detection and aimbot system.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import GameDetector
from ui import GameDetectorUI

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import cv2
        import torch
        import ultralytics
        import supervision
        import mss
        import win32api
        import win32con
        from PIL import Image, ImageTk
        return True
    except ImportError as e:
        return False, str(e)

def check_model_file():
    """Check if the model file exists"""
    # Get the directory where main.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "Models", "best.pt")
    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}"
    return True, model_path

def main():
    """Main application entry point"""
    print("Starting Project IGI Game Application...")
    
    # Check dependencies
    print("Checking dependencies...")
    deps_check = check_dependencies()
    if deps_check is not True:
        print(f"Missing dependencies: {deps_check[1]}")
        messagebox.showerror("Missing Dependencies", 
                           f"Please install required packages:\n{deps_check[1]}")
        return
    
    # Check model file
    print("Checking model file...")
    model_check = check_model_file()
    if not model_check[0]:
        print(f"Model error: {model_check[1]}")
        messagebox.showerror("Model Error", model_check[1])
        return
    
    model_path = model_check[1]
    print(f"Model found: {model_path}")
    
    try:
        # Initialize detector
        print("Initializing detector...")
        detector = GameDetector(model_path)
        
        # Initialize and run UI
        print("Starting UI...")
        app = GameDetectorUI(detector)
        app.run()
        
    except Exception as e:
        print(f"Application error: {str(e)}")
        messagebox.showerror("Application Error", f"Failed to start application:\n{str(e)}")
    
    finally:
        print("Application closed.")

if __name__ == "__main__":
    main()

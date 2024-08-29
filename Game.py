from ultralytics import YOLO
import torch

from utils import Run_model_video
from utils import Run_Model_Image


model = YOLO(r'D:\Github\AI\Computer Vision\Object Detection Project Igi 1\Models\kaggle\working\runs\detect\train\weights\best.pt').to("cuda")

video_path = r"Test Video_and_image\test_video.mp4"
img_path = r"Test Video_and_image\test_image.jpg"


# Run_model_video(model , video_path)
Run_Model_Image(model , img_path)


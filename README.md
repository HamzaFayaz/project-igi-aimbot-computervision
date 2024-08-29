
# Object Detection and AI Control in Project IGI 1



This project implements advanced object detection and AI-based control for the Project IGI 1 game. It demonstrates object detection using YOLO models, real-time video processing, and future plans for integrating reinforcement learning to control the game.

## Requirements

To run this project, you'll need to install the following packages:

- **PyTorch**: A deep learning framework with CUDA support for GPU acceleration.
- **YOLO**: Object detection models from Ultralytics.
- **OpenCV**: For image and video handling.

You can install the required packages using the following commands:

```bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python-headless

```

## Demo Video
https://github.com/user-attachments/assets/7d9a0277-2cc9-4f23-8214-6335a30173c1


## Setup
Clone the Repository
Install the Required Packages

## Training the Model
Get the data set from my Roboflow Account : https://universe.roboflow.com/computer-vison-fznpi/project-igi-1/dataset/6


## OR Want the Model 

Get the model from this link :  https://drive.google.com/file/d/1iIR_JXK0uTY0SexgU1i24WVsUaVa1ilM/view?usp=sharing

## How to Use the Model
 # Load the Model
Ensure the trained model is in the correct directory and load it using the provided scripts or notebook.

# Run Inference

Use the inference script or notebook to apply the model to new video inputs. Follow the instructions in train.ipynb in (# model folder) to process the video and visualize the results.
Review Results
Check the generated video output (result_test.mp4) to review the model's performance and object detection results.



## Advancements
TensorRT Integration: TensorRT is used to optimize inference on the GPU, providing improved FPS and performance. This integration is planned for enhancing real-time processing capabilities.



## Future Development
## Reinforcement Learning Model: A reinforcement learning model well be trained to aim and fire at enemies. This model represents an initial step towards more sophisticated AI controls for the game.
Game Control with AI: Plans are underway to develop a comprehensive AI system to control the overall game. This includes managing game dynamics and strategic decisions through AI algorithms.



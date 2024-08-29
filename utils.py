import cv2
import time

def Run_model_video(model, video_path, frame_skip_number=2,conf_score=0.5):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get the original frame rate of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")

    # Initialize a variable to keep track of the previous time
    previous_time = time.time()

    # Initialize variables to store previous bounding boxes, scores, and classes
    previous_boxes = []
    previous_scores = []
    previous_classes = []

    # Loop through the video frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1000, 600))

        # Process only every nth frame based on frame_skip_number
        if frame_count % frame_skip_number == 0:
            # Perform inference on the frame
            results = model.predict(frame, conf=conf_score)

            # Clear previous boxes and store current results
            previous_boxes = []
            previous_scores = []
            previous_classes = []

            # Draw the results on the frame and store them for future use
            for result in results:
                for box in result.boxes:
                    # Get the box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    previous_boxes.append((x1, y1, x2, y2))

                    # Get the confidence score and class name
                    score = box.conf[0]
                    previous_scores.append(score)

                    class_name = model.names[int(box.cls[0])]
                    previous_classes.append(class_name)

                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw the class name and confidence score
                    cv2.putText(frame, f'{class_name} {score:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Draw the previous bounding boxes on the current frame
            for i in range(len(previous_boxes)):
                x1, y1, x2, y2 = previous_boxes[i]
                score = previous_scores[i]
                class_name = previous_classes[i]

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw the class name and confidence score
                cv2.putText(frame, f'{class_name} {score:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate and display the current FPS
        current_time = time.time()
        try:
            fps = 1 / (current_time - previous_time)
        except ZeroDivisionError:
            fps = 0
            print("Error: Division by zero occurred")
        previous_time = current_time

        # Display the FPS on the frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the frame with the detections
        cv2.imshow('YOLOv8 Detections', frame)

        frame_count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# Run_model(model, 'path/to/video.mp4', 2)



def Run_Model_Image(model , img_path, conf_score=0.5):
    
# Make predictions on the image
    img = cv2.imread(img_path)
    results = model(img_path,verbose = True,conf=conf_score)
# results = model.infer(image="YOUR_IMAGE.jpg")
    results = results[0]
# Convert the image from BGR to RGB (since OpenCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Iterate over the detected boxes and draw them on the image
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        cls = int(box.cls[0])  # Class ID
        conf = box.conf[0]  # Confidence score
    # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box with thickness 2

    # Draw the label with class name and confidence
        label = f"{results.names[cls]} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    img = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
    cv2.imshow("Result ",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
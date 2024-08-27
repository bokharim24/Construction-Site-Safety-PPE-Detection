import cv2
from ultralytics import YOLO

# Load the YOLOv8s model
model = YOLO('yolov10.pt')  # You can replace 'yolov8s.pt' with any model variant

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break
    
    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Annotate the frame with the detection results
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Webcam', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
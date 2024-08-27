import cv2
from ultralytics import YOLO

# Load the YOLOv8 model locally
model = YOLO('/Users/danishbokhari/testinggithub/Construction-Site-Safety-PPE-Detection/models/finalTest.pt')

model.predict(source = 0, show = True)

# # Access your webcam feed
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run inference on the webcam frame
#     results = model(frame)

#     # Loop through detections and draw bounding boxes
#     for result in results:
#         boxes = result.boxes  # Get bounding box outputs
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert tensor to int
#             class_name = model.names[int(box.cls[0])]  # Get class name
#             conf = box.conf.item()  # Get confidence score

#             # Draw the bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             label = f'{class_name} {conf:.2f}'
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     # Show the frame
#     cv2.imshow('YOLOv8 Detection', frame)

#     # Break loop with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
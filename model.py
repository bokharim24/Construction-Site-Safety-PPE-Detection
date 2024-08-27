import cv2
from flask import Flask, render_template, Response,jsonify
from ultralytics import YOLO
import time


app = Flask(__name__)

#0: 'Boots', 1: 'Gloves', 2: 'Goggles', 3: 'Hardhat', 4: 'Mask', 5: 'NO-Boots', 6: 'NO-Gloves', 
# 7: 'NO-Goggles', 8: 'NO-Hardhat', 9: 'NO-Mask', 10: 'NO-Safety Vest', 11: 'Person', 12: 'Safety Cone', 
# 13: 'Safety Vest', 14: 'machinery', 15: 'vehicle'}

# Load the trained YOLO model
model = YOLO("/Users/danishbokhari/testinggithub/Construction-Site-Safety-PPE-Detection/models/epoch100.pt")

# camera_url = 'rtsp://admin:amcrest123@192.168.1.18:554/cam/realmonitor?channel=1&subtype=0'

# Open the video file using OpenCV
cap = cv2.VideoCapture(0)

# # Set resolution, FPS, and buffer size
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



GOGGLES_CLASS_ID = 0
NO_GOGGLES_CLASS_ID = 3

color = "gray" 
ppe_detected = None

#Counters for tracking people with and without PPE
people_with_ppe = 0
people_without_ppe = 0

# Timing control to avoid counting the same person multiple times
last_ppe_detected_time = 0
last_no_ppe_detected_time = 0
cooldown_time = 2  # in seconds
fps = cap.get(cv2.CAP_PROP_FPS)

# Function to generate video frames for the Flask web app
def generate_frames():
    global goggles_seconds, frame_counter, color,ppe_detected,people_with_ppe, people_without_ppe, last_ppe_detected_time, last_no_ppe_detected_time,fps
    zoom_factor = 1.2
    while True:

        ret, frame = cap.read()
        if not ret:
            break
        print(f"Camera FPS: {fps}")
        # Run YOLO inference on the current frame
        frame = zoom_in(frame, zoom_factor=zoom_factor)

        results = model.predict(frame)

        goggles_detected = False
        no_goggles_detected = False
        # Process results
        for result in results:
            boxes = result.boxes  # Get bounding box outputs
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract box coordinates
                conf = box.conf.item()  # Convert confidence tensor to a float
                class_id = int(box.cls[0])  # Class ID
                class_name = model.names[class_id]

                if class_id == GOGGLES_CLASS_ID:
                    goggles_detected = True
                elif class_id == NO_GOGGLES_CLASS_ID:
                    no_goggles_detected = True
                
                # Draw the bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class: {class_name}, Conf: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
      
       
       
        current_time = time.time()

        if goggles_detected:
            color = "green"
            if current_time - last_ppe_detected_time > cooldown_time:  # Check cooldown
                people_with_ppe += 1
                last_ppe_detected_time = current_time  # Update last detection time
        elif no_goggles_detected:
            color = "red"
            if current_time - last_no_ppe_detected_time > cooldown_time:  # Check cooldown
                people_without_ppe += 1
                last_no_ppe_detected_time = current_time  # Update last detection time
        else:
            color = "gray"  # No goggles or no-goggles detected

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to the web browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def zoom_in(img, zoom_factor=1.5):
    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Define the new boundaries after zooming in
    new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)
    
    left = center_x - new_width // 2
    right = center_x + new_width // 2
    top = center_y - new_height // 2
    bottom = center_y + new_height // 2

    # Crop the image
    cropped_img = img[top:bottom, left:right]

    # Resize back to the original size
    zoomed_img = cv2.resize(cropped_img, (width, height))

    return zoomed_img
# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')  # HTML page to display video feed

@app.route('/ppe')
def ppe():
    global color
    return render_template('ppe.html',color = color)  # HTML page to display video feed

# Route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_color')
def get_color():
    return jsonify({'color': color})

# Route to get the count of people with and without PPE
@app.route('/get_ppe_counts')
def get_ppe_counts():
    return jsonify({'people_with_ppe': people_with_ppe, 'people_without_ppe': people_without_ppe})


# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
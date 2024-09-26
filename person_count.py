import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Get the output layer indices
output_layer_indices = net.getUnconnectedOutLayers()
if isinstance(output_layer_indices, np.ndarray):
    output_layers = [layer_names[i - 1] for i in output_layer_indices.flatten()]
else:
    output_layers = [layer_names[output_layer_indices - 1]]


# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load video
cap = cv2.VideoCapture("2.mp4")

# Initialize count
people_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape
    people_count = 0

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to show on the screen
    class_ids = []
    confidences = []
    boxes = []

    # Loop through each output layer
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold for detection
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maxima Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Count people and draw boxes
    if len(indexes) > 0:
        for i in indexes.flatten():
            if class_ids[i] == 0:  # Class ID 0 corresponds to 'person'
                people_count += 1
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])  # Define the classes list
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{label}: {confidences[i]:.2f}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display count on the frame
    cv2.putText(frame, f'People Count: {people_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

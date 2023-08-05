import cv2
import numpy as np

def load_yolo():
    # Load the pre-trained YOLO model and classes
    net = cv2.dnn_DetectionModel('yolov3.cfg', 'yolov3.weights')
    net.setInputSize(416, 416)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    return net, classes


def detect_objects(frame, net, classes, confidence_threshold=0.5, nms_threshold=0.4):
    # Perform object detection
    class_ids, confidences, boxes = net.detect(frame, nmsThreshold=nms_threshold)

    # Convert class_ids and confidences to flat NumPy arrays
    class_ids = np.array(class_ids).flatten()
    confidences = np.array(confidences).flatten()

    # Convert boxes to a list
    boxes = list(boxes)

    # Filter detections based on confidence threshold
    indices = [i for i, conf in enumerate(confidences) if conf > confidence_threshold]
    filtered_class_ids = class_ids[indices]
    filtered_confidences = confidences[indices]
    filtered_boxes = [boxes[i] for i in indices]

    # Draw bounding boxes and labels
    font = cv2.FONT_HERSHEY_PLAIN
    objects_detected = len(filtered_boxes)
    for i in range(objects_detected):
        x, y, w, h = filtered_boxes[i]
        class_id = filtered_class_ids[i]
        confidence = filtered_confidences[i]

        # Get the corresponding class name
        class_name = classes[class_id]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x, y - 10), font, 1, (0, 255, 0), 1)

    return frame, objects_detected


def main():
    net, classes = load_yolo()

    # Set up the video capture
    cap = cv2.VideoCapture(0)  # Change to 0 for webcam or provide a video file path

    # Loop through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, objects_detected = detect_objects(frame, net, classes)

        # Display the resulting frame
        cv2.imshow('Live Object Detection', frame)

        # Check for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

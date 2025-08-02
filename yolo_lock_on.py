import socket
import cv2
import numpy as np
import zmq
import time
from ultralytics import YOLO

# This script runs on a powerful machine (e.g., Mac) to handle the computationally
# intensive parts: receiving the video stream, running YOLO inference, and
# sending back tracking commands.

# === YOLO Model Setup ===
# Load a pre-trained YOLOv8 nano model for fast object detection
model = YOLO("yolov8n.pt")

# === TCP Socket to Pi (Client Side) ===
TCP_IP = "raspberrypi.local"  # Hostname of the Raspberry Pi
TCP_PORT = 8485
# Create a TCP socket and connect to the Pi's server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((TCP_IP, TCP_PORT))

# Helper function to receive a specific number of bytes from the socket
def recv_exact(sock, size):
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            # Connection was closed before all data was received
            return None
        data += packet
    return data

# === ZMQ Sender to Pi (Publisher) ===
# We use ZeroMQ to send commands back to the Pi because it's a fast
# and efficient messaging library, and the Pi will be listening on a
# separate subscriber socket.
context = zmq.Context()
zmq_socket = context.socket(zmq.PUB)
# The Mac acts as the publisher, connecting to the Pi's address
zmq_socket.connect("tcp://raspberrypi.local:5555")

# === Tracking State Variables ===
locked_box = None      # Stores the bounding box of the currently locked person
lock_active = False    # Flag to indicate if a lock-on is active
results = None         # Stores the latest YOLO detection results
fps = 0                # Frames per second counter
prev_time = time.time()  # Timestamp for FPS calculation

# Helper function to get the center point (x, y) of a bounding box
def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# Helper function to compute the Intersection over Union (IoU) of two bounding boxes
def compute_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# Mouse click event handler for OpenCV window
def click_event(event, x, y, flags, param):
    global locked_box, lock_active, results
    # Check for a left mouse button click and if YOLO results are available
    if event == cv2.EVENT_LBUTTONDOWN and results:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        classes = results[0].boxes.cls.cpu().numpy() # Class IDs
        best_box = None
        min_dist = float("inf")

        # Iterate through all detected objects
        for i, box in enumerate(boxes):
            # We are only interested in "person" detections (class ID 0)
            if int(classes[i]) != 0:
                continue

            cx, cy = get_center(box)
            # Calculate the distance from the click to the center of the box
            dist = np.linalg.norm([x - cx, y - cy])
            # If this box is closer than the previous best and within a certain threshold
            if dist < min_dist and dist < 100:
                best_box = box
                min_dist = dist
        
        # If a person was found near the click location, lock onto them
        if best_box is not None:
            locked_box = best_box
            lock_active = True

# Create a display window and set the mouse callback function
cv2.namedWindow("Lock-On Human Detection")
cv2.setMouseCallback("Lock-On Human Detection", click_event)

# === Frame Receive + Inference Loop ===
while True:
    try:
        # Receive the 4-byte frame size header
        frame_len_bytes = recv_exact(client_socket, 4)
        if frame_len_bytes is None:
            print("[ERROR] Lost connection")
            break

        # Convert the byte header to an integer frame size
        frame_len = int.from_bytes(frame_len_bytes, byteorder='big')
        # Receive the full frame data
        frame_data = recv_exact(client_socket, frame_len)
        if frame_data is None:
            print("[ERROR] Incomplete frame")
            break

        # Decode the JPEG frame data back into an OpenCV image
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print("[WARN] Empty frame received. Skipping...")
            continue

        # Run YOLO inference on the received frame.
        # `classes=[0]` restricts detection to only the "person" class.
        results = model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        annotated = frame.copy()

        new_locked_box = None
        max_iou = 0

        # --- Tracking Logic ---
        if lock_active:
            # If a lock is active, find the person with the highest IoU
            # to the previous locked box to maintain the lock.
            for i, box in enumerate(boxes):
                iou = compute_iou(locked_box, box)
                # We need a high IoU to confirm it's the same person
                if iou > max_iou and iou > 0.4:
                    max_iou = iou
                    new_locked_box = box

            # If a person was successfully re-identified, update the lock
            if new_locked_box is not None:
                locked_box = new_locked_box
                x1, y1, x2, y2 = map(int, locked_box)
                cx, cy = get_center(locked_box)
                w = x2 - x1
                h = y2 - y1

                # Send the tracking data (center and size) to the Pi via ZMQ
                # This data will be used for flight control.
                zmq_socket.send_string(f"{cx},{cy},{w},{h}")

                # Draw a red rectangle for the locked person
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, "LOCKED", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # If no person with a high enough IoU was found, lose the lock
                lock_active = False
                locked_box = None
                print("[WARN] Lost track of target.")

        # --- Display Logic ---
        # Draw green bounding boxes for all other detected people
        for i, box in enumerate(boxes):
            # Skip drawing a green box if this is the currently locked person
            if lock_active and compute_iou(locked_box, box) > 0.9:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, "person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show the annotated frame
        cv2.imshow("Lock-On Human Detection", annotated)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"[ERROR] {e}")
        break

# Clean up
client_socket.close()
cv2.destroyAllWindows()

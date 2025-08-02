import socket
import cv2
import numpy as np
import zmq
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

TCP_IP = "raspberrypi.local"  # Or your Pi's IP
TCP_PORT = 8485
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((TCP_IP, TCP_PORT))

def recv_exact(sock, size):
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

# ZMQ setup
context = zmq.Context()
zmq_socket = context.socket(zmq.PUB)
zmq_socket.connect("tcp://raspberrypi.local:5555")

locked_box = None
lock_active = False
lock_lost_counter = 0
results = None
fps = 0
prev_time = time.time()

def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def click_event(event, x, y, flags, param):
    global locked_box, lock_active, results
    if event == cv2.EVENT_LBUTTONDOWN and results:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        best_box = None
        min_dist = float("inf")
        for i, box in enumerate(boxes):
            if int(classes[i]) != 0:
                continue
            cx, cy = get_center(box)
            dist = np.linalg.norm([x - cx, y - cy])
            if dist < min_dist and dist < 100:
                best_box = box
                min_dist = dist
        if best_box is not None:
            locked_box = best_box
            lock_active = True

cv2.namedWindow("Lock-On Human Detection")
cv2.setMouseCallback("Lock-On Human Detection", click_event)

while True:
    try:
        frame_len_bytes = recv_exact(client_socket, 4)
        if frame_len_bytes is None:
            print("[ERROR] Lost connection")
            break

        frame_len = int.from_bytes(frame_len_bytes, byteorder='big')
        frame_data = recv_exact(client_socket, frame_len)
        if frame_data is None:
            print("[ERROR] Incomplete frame")
            break

        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        results = model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        annotated = frame.copy()

        new_locked_box = None
        max_iou = 0

        if lock_active:
            for i, box in enumerate(boxes):
                if int(classes[i]) != 0:
                    continue
                iou = compute_iou(locked_box, box)
                cx_old, cy_old = get_center(locked_box)
                cx_new, cy_new = get_center(box)
                dist = np.linalg.norm([cx_old - cx_new, cy_old - cy_new])

                if (iou > 0.3 or dist < 80) and iou > max_iou:
                    max_iou = iou
                    new_locked_box = box

            if new_locked_box is not None:
                locked_box = new_locked_box
                lock_lost_counter = 0

                x1, y1, x2, y2 = map(int, locked_box)
                cx, cy = get_center(locked_box)
                w = x2 - x1
                h = y2 - y1

                zmq_socket.send_string(f"{cx},{cy},{w},{h}")
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, "LOCKED", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                lock_lost_counter += 1
                if lock_lost_counter > 5:
                    locked_box = None
                    lock_active = False
                    lock_lost_counter = 0

        for i, box in enumerate(boxes):
            if lock_active and compute_iou(locked_box, box) > 0.9:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, "person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Lock-On Human Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("[ERROR]", str(e))
        break

client_socket.close()
cv2.destroyAllWindows()

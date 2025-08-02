import socket
import time
import cv2
from picamera2 import Picamera2
import numpy as np

HOST = ''  # Listen on all interfaces
PORT = 8485

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()

print("[INFO] Waiting for connection...")

while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        conn, addr = s.accept()
        print(f"[INFO] Connection from {addr}")

        try:
            while True:
                frame = picam2.capture_array()
                _, jpeg = cv2.imencode(".jpg", frame)
                data = jpeg.tobytes()

                # Send frame size first (4 bytes)
                conn.sendall(len(data).to_bytes(4, byteorder='big'))
                conn.sendall(data)
                time.sleep(0.03)  # ~30 FPS
        except (BrokenPipeError, ConnectionResetError):
            print("[WARN] Connection closed by client. Waiting again...")
            conn.close()
            continue
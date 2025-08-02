import socket
import time
import cv2
from picamera2 import Picamera2
import numpy as np

# This script runs on the Raspberry Pi. It captures video from the Picamera2
# and streams it over a TCP socket to a client (the Mac) for processing.

# --- TCP Server Setup ---
HOST = ''  # Listen on all available network interfaces
PORT = 8485  # Port for the video stream

# --- Camera Setup ---
picam2 = Picamera2()
# Configure the camera's main stream for a 640x480 resolution
picam2.preview_configuration.main.size = (640, 480)
# Set the pixel format to BGR888 (compatible with OpenCV)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()

print("[INFO] Waiting for connection...")

# --- Main Server Loop ---
while True:
    # Create a new socket for each connection attempt.
    # This ensures the server can handle new connections if a client disconnects.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Allow the socket to be reused immediately after it's closed
        # This prevents "Address already in use" errors.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind the socket to the host and port
        s.bind((HOST, PORT))
        # Listen for a single incoming connection
        s.listen(1)
        # Accept the connection from the client
        conn, addr = s.accept()
        print(f"[INFO] Connection from {addr}")

        try:
            # --- Frame Streaming Loop ---
            while True:
                # Capture a frame from the camera as a NumPy array
                frame = picam2.capture_array()
                # Encode the frame into JPEG format to reduce data size
                _, jpeg = cv2.imencode(".jpg", frame)
                # Convert the encoded frame to a byte array
                data = jpeg.tobytes()

                # Send the size of the frame data (4 bytes) before the frame itself.
                # This allows the client to know how much data to expect for the frame.
                conn.sendall(len(data).to_bytes(4, byteorder='big'))
                # Send the actual frame data
                conn.sendall(data)
                # Introduce a small delay to control the frame rate (~30 FPS)
                time.sleep(0.03)
        
        except (BrokenPipeError, ConnectionResetError):
            # This handles cases where the client closes the connection unexpectedly
            print("[WARN] Connection closed by client. Waiting again...")
            conn.close()
            continue

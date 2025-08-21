# Lock-On Drone Autopilot

This project implements a distributed computer vision system for a drone, allowing it to autonomously track a human target. The system leverages a powerful host machine (e.g., a Mac or PC) to perform computationally intensive object detection and a lightweight Raspberry Pi to handle camera streaming and drone control.

## Key Features

* **Distributed Architecture**: The system is split into three main components:
    * **Raspberry Pi Streamer (`streamer.py`)**: Streams raw video from a Picamera2 to a host machine.
    * **Host Machine Detector (`yolo_lock_on.py`)**: Receives the video, performs real-time human detection using YOLOv8, and allows a user to "lock on" to a specific person via a mouse click. It then sends tracking commands (target's position and size) back to the Pi.
    * **Raspberry Pi Autopilot (`zmq_listener_mavlink.py`)**: Receives tracking data and translates it into MAVLink commands to control the drone's yaw and forward velocity via a flight controller (e.g., Pixhawk).
* **Efficient Communication**:
    * Video streaming is handled via a **TCP socket** for reliable, ordered delivery of frames.
    * Tracking commands are sent back to the Pi using a **ZeroMQ (ZMQ)** publish-subscribe pattern, ensuring low-latency communication for control signals.
* **Real-Time Tracking**: The system uses the Intersection over Union (IoU) metric to maintain a lock on the target even if they are temporarily occluded or move quickly.

## Project Structure

* `streamer.py`: Runs on the Raspberry Pi. This is the video streaming server.
* `yolo_lock_on.py`: Runs on the host machine (e.g., Mac/PC). This is the YOLO inference and locking client.
* `zmq_listener_mavlink.py`: Runs on the Raspberry Pi. This is the MAVLink autopilot controller.
* `README.md`: This file.

## Requirements

### Hardware

* **Drone**: A drone with a compatible flight controller (e.g., Pixhawk with ArduPilot/PX4 firmware).
* **Companion Computer**: A Raspberry Pi with a Picamera2 module.
* **Host Machine**: A powerful computer (e.g., Mac, Windows, Linux), preferably with a GPU for fast YOLO inference.
* **Network**: A stable WiFi network connecting the Raspberry Pi and the host machine.

### Software

* **On the Raspberry Pi:**
    * Python 3.x
    * `picamera2`
    * `opencv-python`
    * `numpy`
    * `pyzmq`
    * `pymavlink`
* **On the Host Machine:**
    * Python 3.x
    * `ultralytics` (for YOLOv8)
    * `opencv-python`
    * `numpy`
    * `pyzmq`

## Setup and Usage

### 1. Raspberry Pi Setup

1.  **Clone the repository.**
2.  **Install dependencies**:
    ```bash
    pip install picamera2 opencv-python numpy pyzmq pymavlink
    ```
3.  **Configure MAVLink**:
    * Ensure your flight controller is connected to the Pi's serial port (e.g., `/dev/serial0`).
    * Configure your flight controller (e.g., with Mission Planner or QGroundControl) to accept MAVLink commands from the serial port.
    * Set your flight mode to `GUIDED` or a similar mode that allows for velocity commands.

### 2. Host Machine Setup

1.  **Clone the repository.**
2.  **Install dependencies**:
    ```bash
    pip install ultralytics opencv-python numpy pyzmq
    ```
3.  **Find your Raspberry Pi's IP/hostname**:
    * You can often use `raspberrypi.local` or find its IP address on your network.
    * Make sure to update the `TCP_IP` and ZMQ connection string in `yolo_lock_on.py` to match the Pi's address.

### 3. Running the System

1.  **On the Raspberry Pi**, start the streamer and the autopilot script in two separate terminal windows:
    * **Terminal 1**: `python3 streamer.py`
    * **Terminal 2**: `python3 zmq_listener_mavlink.py`
2.  **On the host machine**, start the client script:
    * `python3 yolo_lock_on.py`

You should see a window pop up on your host machine showing the video feed from the Pi. YOLO bounding boxes will appear around detected people. Click on a person to initiate the "lock-on" sequence. The drone will then attempt to follow the person by adjusting its yaw and forward velocity. Press `q` on the host machine to quit.

## Troubleshooting

* **Connection Refused error**: Make sure `streamer.py` is running on the Pi and that the `TCP_IP` in `yolo_lock_on.py` is correct.
* **No MAVLink heartbeat**: Check your serial connection to the flight controller and ensure it's powered on and properly configured.
* **Slow FPS**: Ensure your host machine has a capable GPU and that `ultralytics` is configured to use it.
* **Erratic Drone Control**: You may need to adjust the `YAW_GAIN` and `FORWARD_GAIN` constants in `zmq_listener_mavlink.py` to better suit your drone's characteristics. Start with small values and increase them incrementally.

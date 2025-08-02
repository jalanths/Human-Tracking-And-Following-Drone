import zmq
from pymavlink import mavutil
import time

# This script runs on the Raspberry Pi. It receives tracking data
# from the Mac over ZMQ and converts it into MAVLink commands to
# control a drone (e.g., connected to a Pixhawk flight controller).

# === Setup MAVLink to Pixhawk ===
# Connect to the Pixhawk flight controller over a serial port
master = mavutil.mavlink_connection('/dev/serial0', baud=57600)
print("[INFO] Waiting for heartbeat from Pixhawk...")
# Wait for the first heartbeat message to confirm connection
master.wait_heartbeat()
print(f"[INFO] Heartbeat received from system {master.target_system}")

# === Setup ZMQ to receive tracking data from Mac ===
context = zmq.Context()
# Create a subscriber socket to receive data from the Mac
socket = context.socket(zmq.SUB)
# Bind the socket to all interfaces on port 5555. The Mac will connect to this.
socket.bind("tcp://*:5555")
# Subscribe to all messages (an empty string subscribes to all topics)
socket.setsockopt_string(zmq.SUBSCRIBE, "")
print("[ZMQ] Listening for tracking data on tcp://*:5555")

# === MAVLink Time Tracking ===
# Record the script's start time to calculate the `time_boot_ms` for MAVLink
start_time = time.time()

# === Control Tuning Constants ===
# These constants are used to translate pixel data into control signals.
# You will likely need to tune these for your specific drone and camera setup.
CENTER_TOLERANCE = 50   # If the person is within this many pixels of the center, don't yaw.
MAX_SPEED = 1.5         # Maximum forward/backward speed (in m/s)
YAW_GAIN = 0.002        # Controls how aggressively the drone yaws to center the person (rad/s per pixel offset)
FORWARD_GAIN = 0.005    # Controls how fast the drone moves forward/backward based on the person's size (m/s per area unit)

# === Function to send velocity and yaw rate to the drone ===
def send_velocity(vx, vy, vz, yaw_rate):
    # MAVLink requires the time since the boot of the MAVLink system in milliseconds.
    time_boot_ms = int((time.time() - start_time) * 1000)
    # Fallback for integer overflow on very long runtimes
    if time_boot_ms < 0 or time_boot_ms > 4294967295:
        time_boot_ms = 0

    # This is the main MAVLink command to control the drone.
    master.mav.set_position_target_local_ned_send(
        time_boot_ms,                      # time since boot
        master.target_system,              # Target system ID
        master.target_component,           # Target component ID
        mavutil.mavlink.MAV_FRAME_BODY_NED, # Frame of reference (body-fixed)
        0b0000111111000111,                # Type mask: use velocity & yaw_rate only
        0, 0, 0,                           # Position (unused)
        vx, vy, vz,                        # Velocity in X, Y, Z axes (m/s)
        0, 0, 0,                           # Acceleration (unused)
        0,                                 # Yaw (unused)
        yaw_rate                           # Yaw rate in radians per second
    )

print("[READY] Tracking → MAVLink command loop started")

# === Main Autopilot Loop ===
while True:
    try:
        # Receive the tracking data string from the Mac, e.g., "320,240,100,150"
        msg = socket.recv_string()
        cx, cy, w, h = map(int, msg.strip().split(","))
        frame_w, frame_h = 640, 480

        # Calculate the horizontal error (difference from the center of the frame)
        error_x = cx - frame_w // 2
        # Calculate the area of the bounding box
        area = w * h

        # --- Yaw control ---
        # The drone will yaw based on how far the target is from the center.
        # A negative error_x means the target is on the left, so we need to yaw left (negative yaw rate).
        yaw_rate = -error_x * YAW_GAIN

        # --- Forward/Backward control ---
        # The drone will move forward to a target area (15000 is a target area).
        # If the current area is smaller than the target, move forward.
        # If the current area is larger, move backward.
        forward_speed = (15000 - area) * FORWARD_GAIN
        # Clamp the speed to prevent excessively fast movement
        forward_speed = max(min(forward_speed, MAX_SPEED), -MAX_SPEED)

        # Send the calculated velocities and yaw rate to the drone
        send_velocity(forward_speed, 0, 0, yaw_rate)

        # Print the tracking status for debugging
        print(f"[TRACK] cx={cx} area={area} vx={forward_speed:.2f} yaw={yaw_rate:.3f}")

    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(0.5)

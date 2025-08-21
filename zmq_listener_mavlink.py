import zmq
from pymavlink import mavutil
import time

# === Setup MAVLink to Pixhawk ===
master = mavutil.mavlink_connection('/dev/serial0', baud=57600)
print("[INFO] Waiting for heartbeat from Pixhawk...")
master.wait_heartbeat()
print(f"[INFO] Heartbeat received from system {master.target_system}")

# === Setup ZMQ to receive tracking data from Mac ===
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.bind("tcp://*:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")
print("[ZMQ] Listening for tracking data on tcp://*:5555")

# === Time boot tracking for MAVLink (needs 32-bit ms uptime) ===
start_time = time.time()

# === Control tuning constants ===
CENTER_TOLERANCE = 50   # pixels
MAX_SPEED = 1.5         # m/s forward
YAW_GAIN = 0.002        # radians per pixel offset
FORWARD_GAIN = 0.005    # m/s per area unit

# === Function to send velocity + yaw rate to drone ===
def send_velocity(vx, vy, vz, yaw_rate):
    time_boot_ms = int((time.time() - start_time) * 1000)
    if time_boot_ms < 0 or time_boot_ms > 4294967295:
        time_boot_ms = 0  # fallback in rare edge case

    master.mav.set_position_target_local_ned_send(
        time_boot_ms,                      # time_boot_ms (ms since script started)
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,                # enable only velocity + yaw_rate
        0, 0, 0,                           # x, y, z positions (unused)
        vx, vy, vz,                        # x, y, z velocity
        0, 0, 0,                           # acceleration (not used)
        0,                                 # yaw (absolute) not used
        yaw_rate                           # yaw_rate in rad/s
    )

print("[READY] Tracking → MAVLink command loop started")

# === Main loop ===
while True:
    try:
        msg = socket.recv_string()  # from Mac: "cx,cy,w,h"
        cx, cy, w, h = map(int, msg.strip().split(","))
        frame_w, frame_h = 640, 480

        error_x = cx - frame_w // 2
        area = w * h

        # Yaw control (turn left/right)
        yaw_rate = -error_x * YAW_GAIN

        # Forward control (closer bbox = slow down)
        forward_speed = (15000 - area) * FORWARD_GAIN
        forward_speed = max(min(forward_speed, MAX_SPEED), -MAX_SPEED)

        send_velocity(forward_speed, 0, 0, yaw_rate)

        print(f"[TRACK] cx={cx} area={area} vx={forward_speed:.2f} yaw={yaw_rate:.3f}")

    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(0.5)
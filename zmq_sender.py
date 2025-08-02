import zmq

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")  # Mac is the ZMQ server
print("[ZMQ] Publisher bound to tcp://*:5556")

def send_lock_data(box, frame_shape):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    width = int(x2 - x1)
    height = int(y2 - y1)

    frame_h, frame_w, _ = frame_shape
    data = {
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "frame_w": frame_w,
        "frame_h": frame_h
    }

    socket.send_json(data)

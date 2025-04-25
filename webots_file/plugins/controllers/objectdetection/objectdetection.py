from controller import Robot, Keyboard
from ultralytics import YOLO
import numpy as np
import cv2
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Enable camera
camera = robot.getDevice("CameraTop")
camera.enable(timestep)

# Enable keyboard
keyboard = Keyboard()
keyboard.enable(timestep)

# Load YOLO model
model = YOLO("C:\Users\Hp\Desktop\object_detection_yolo_webots\Object_detection\best.pt")

# Get joints
joints = {
    "LHipPitch": robot.getDevice("LHipPitch"),
    "RHipPitch": robot.getDevice("RHipPitch"),
    "LKneePitch": robot.getDevice("LKneePitch"),
    "RKneePitch": robot.getDevice("RKneePitch"),
    "LAnklePitch": robot.getDevice("LAnklePitch"),
    "RAnklePitch": robot.getDevice("RAnklePitch"),
    "HeadYaw": robot.getDevice("HeadYaw"),
    "HeadPitch": robot.getDevice("HeadPitch")
}

# Set initial joint positions
for joint in joints.values():
    joint.setPosition(0.0)

step = 0  # step counter for movement oscillation

# Movement functions
def walk(step, direction="forward"):
    swing = math.sin(step * 0.1) * 0.3

    if direction == "forward":
        hip_left = swing
        hip_right = -swing
    elif direction == "backward":
        hip_left = -swing
        hip_right = swing
    elif direction == "left":
        hip_left = -swing * 0.5
        hip_right = swing * 0.5
    elif direction == "right":
        hip_left = swing * 0.5
        hip_right = -swing * 0.5
    else:
        hip_left = hip_right = 0

    joints["LHipPitch"].setPosition(hip_left)
    joints["LKneePitch"].setPosition(-hip_left * 0.5)
    joints["LAnklePitch"].setPosition(hip_left * 0.25)

    joints["RHipPitch"].setPosition(hip_right)
    joints["RKneePitch"].setPosition(-hip_right * 0.5)
    joints["RAnklePitch"].setPosition(hip_right * 0.25)

# Main loop
while robot.step(timestep) != -1:
    key = keyboard.getKey()

    # Image and detection
    img = camera.getImage()
    w = camera.getWidth()
    h = camera.getHeight()
    img_np = np.frombuffer(img, np.uint8).reshape((h, w, 4))
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)

    results = model(img_rgb)
    annotated = results[0].plot()
    cv2.imshow("YOLO Detection", annotated)
    cv2.waitKey(1)

    # Movement logic
    if key in [Keyboard.UP, ord('W')]:
        walk(step, "forward")
        step += 1
    elif key in [Keyboard.DOWN, ord('S')]:
        walk(step, "backward")
        step += 1
    elif key in [Keyboard.LEFT, ord('A')]:
        walk(step, "left")
        step += 1
    elif key in [Keyboard.RIGHT, ord('D')]:
        walk(step, "right")
        step += 1
    else:
        step = 0
        for joint in joints.values():
            joint.setPosition(0.0)
# Main loop
while robot.step(timestep) != -1:
    key = keyboard.getKey()

    # Get camera image and convert
    img = camera.getImage()
    w, h = camera.getWidth(), camera.getHeight()
    img_np = np.frombuffer(img, np.uint8).reshape((h, w, 4))
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)

    # Object detection
    results = model(img_rgb, verbose=False)
    annotated = results[0].plot()

    # Display detection results
    cv2.imshow("YOLOv8 Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Keyboard movement handling
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
        # Reset joint positions when no key is pressed
        for joint in joints.values():
            joint.setPosition(0.0)

# Release OpenCV resources properly at the end
cv2.destroyAllWindows()

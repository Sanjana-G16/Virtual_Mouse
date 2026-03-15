import cv2
import pyautogui
import numpy as np

# Set the color range for green color
lower_green = np.array([40, 100, 100])
upper_green = np.array([80, 255, 255])

# Set the webcam
cap = cv2.VideoCapture(0)

# Get the screen size
screen_width, screen_height = pyautogui.size()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours
    for contour in contours:
        # Get the area of the contour
        area = cv2.contourArea(contour)

        # Ignore small contours
        if area > 1000:
            # Get the centroid of the contour
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Map the centroid to the screen coordinates
            screen_x = int(cx * screen_width / frame.shape[1])
            screen_y = int(cy * screen_height / frame.shape[0])

            # Move the mouse to the screen coordinates
            pyautogui.moveTo(screen_x, screen_y)

            # Perform a mouse click
            pyautogui.click()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()



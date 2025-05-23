import cv2
import numpy as np

# Load the virtual background image
background = cv2.imread('sion.jpg')

# Check if the background loaded successfully
if background is None:
    print("Error: Could not load background image 'wallpaper.webp'.")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

print("Initializing camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Resize background to match frame size
    background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # Create mask for white areas
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Optional: clean up the mask (morphological operations)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Invert mask to get the person (non-white areas)
    mask_inv = cv2.bitwise_not(mask)

    # Extract the person from the frame
    person = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Extract background from the virtual background image
    virtual_bg = cv2.bitwise_and(background_resized, background_resized, mask=mask)

    # Combine the person and the virtual background
    final_output = cv2.add(person, virtual_bg)

    # Show the final output
    cv2.imshow("Virtual Background Replacement (White BG)", final_output)

    # Exit on ESC or 'q'
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

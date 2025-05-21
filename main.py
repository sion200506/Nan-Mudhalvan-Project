# Virtual background replacement

import cv2
import numpy as np

background = cv2.imread('wallpaper.webp')

cap = cv2.VideoCapture(0)

print("Initializing camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask_inv = cv2.bitwise_not(mask)

    person = cv2.bitwise_and(frame, frame, mask=mask_inv)

    virtual_bg = cv2.bitwise_and(background_resized, background_resized, mask=mask)

    final_output = cv2.add(person, virtual_bg)

    cv2.imshow("Virtual Background Replacement (White BG)", final_output)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

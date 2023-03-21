# gsRemoval_updated
Find the optimal color range to remove green screen by combining two color spaces i.e. HSV and YCbCr

import cv2
import numpy as np

def nothing(x):
    pass
    
## 1.
```python
# create window with trackbars
cv2.namedWindow('panel')
cv2.createTrackbar('L – h', 'panel', 0, 179, nothing)
cv2.createTrackbar('U – h', 'panel', 179, 179, nothing)
cv2.createTrackbar('L – s', 'panel', 0, 255, nothing)
cv2.createTrackbar('U – s', 'panel', 255, 255, nothing)
cv2.createTrackbar('L – v', 'panel', 0, 255, nothing)
cv2.createTrackbar('U – v', 'panel', 255, 255, nothing)
cv2.createTrackbar('L – Y', 'panel', 0, 255, nothing)
cv2.createTrackbar('U – Y', 'panel', 255, 255, nothing)
cv2.createTrackbar('L – Cb', 'panel', 0, 255, nothing)
cv2.createTrackbar('U – Cb', 'panel', 255, 255, nothing)
cv2.createTrackbar('L – Cr', 'panel', 0, 255, nothing)
cv2.createTrackbar('U – Cr', 'panel', 255, 255, nothing)

# set up video capture
cap = cv2.VideoCapture('green_screen_video.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# load background image
background = cv2.imread('background.png')
background = cv2.resize(background, (frame_width, frame_height))

# create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (frame_width, frame_height))

while True:
    # read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break
    
    # get values from trackbars
    l_h = cv2.getTrackbarPos('L – h', 'panel')
    u_h = cv2.getTrackbarPos('U – h', 'panel')
    l_s = cv2.getTrackbarPos('L – s', 'panel')
    u_s = cv2.getTrackbarPos('U – s', 'panel')
    l_v = cv2.getTrackbarPos('L – v', 'panel')
    u_v = cv2.getTrackbarPos('U – v', 'panel')
    l_y = cv2.getTrackbarPos('L – Y', 'panel')
    u_y = cv2.getTrackbarPos('U – Y', 'panel')
    l_cb = cv2.getTrackbarPos('L – Cb', 'panel')
    u_cb = cv2.getTrackbarPos('U – Cb', 'panel')
    l_cr = cv2.getTrackbarPos('L – Cr', 'panel')
    u_cr = cv2.getTrackbarPos('U – Cr', 'panel')
    
    # convert frame to hsv and ycbcr color spaces
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # create masks using trackbar values
    hsv_mask = cv2.inRange(hsv, (l
 ```

## 2.
```python
To combine the masks for both HSV and YCbCr color spaces, you can use the bitwise AND operation as follows:

makefile
Copy code
# combine HSV and YCbCr masks
combined_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

# apply the combined mask to the original frame
masked_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)
Here, combined_mask is the result of applying the bitwise AND operation to hsv_mask and ycrcb_mask. The cv2.bitwise_and() function takes the two input masks as arguments and returns a new mask where only the pixels that are non-zero in both input masks are non-zero. This ensures that only the pixels that satisfy both the HSV and YCbCr color space criteria are kept in the final mask.

Finally, you can apply the combined mask to the original frame using the cv2.bitwise_and() function with the mask argument set to combined_mask. This will apply the mask to the frame and produce the final masked frame where the green screen is removed and replaced with transparency.

To replace the transparent area with a background image, you can use the cv2.cvtColor() function to convert the masked frame to the RGBA color space, which includes an alpha channel for transparency. Then, you can overlay the background image on the masked frame using the alpha channel as a mask. Here is the complete code:

python
Copy code
import cv2
import numpy as np

# define the range of HSV and YCbCr values for green screen
hsv_lower = np.array([82, 80, 52])
hsv_upper = np.array([255, 255, 179])
ycrcb_lower = np.array([0, 135, 85])
ycrcb_upper = np.array([255, 180, 135])

# load the background image
bg_image = cv2.imread('background.png')

# create a video capture object
cap = cv2.VideoCapture('video.mp4')

# create a window to display the masked video
cv2.namedWindow('Masked Video')

# create trackbars for HSV lower and upper values
cv2.createTrackbar('H – L', 'Masked Video', hsv_lower[0], 179, lambda x: None)
cv2.createTrackbar('S – L', 'Masked Video', hsv_lower[1], 255, lambda x: None)
cv2.createTrackbar('V – L', 'Masked Video', hsv_lower[2], 255, lambda x: None)
cv2.createTrackbar('H – U', 'Masked Video', hsv_upper[0], 179, lambda x: None)
cv2.createTrackbar('S – U', 'Masked Video', hsv_upper[1], 255, lambda x: None)
cv2.createTrackbar('V – U', 'Masked Video', hsv_upper[2], 255, lambda x: None)

# create trackbars for YCbCr lower and upper values
cv2.createTrackbar('Y – L', 'Masked Video', ycrcb_lower[0], 255, lambda x: None)
cv2.createTrackbar('Cb – L', 'Masked Video', ycrcb_lower[1], 255, lambda x: None)
cv2.createTrackbar('Cr – L', 'Masked Video', ycrcb_lower[2], 255, lambda x: None)
cv2.createTrackbar('Y – U', 'Masked Video', ycrcb_upper[0], 255, lambda x: None)
cv2.createTrackbar('Cb – U', 'Masked Video', ycrcb_upper[

```

## 3.
```python
import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture('video.mp4')

# Load the background image
background = cv2.imread('background.png')

# Define the lower and upper bounds of the green screen color in HSV color space
lower_green_hsv = np.array([30, 60, 80])
upper_green_hsv = np.array([90, 255, 255])

# Define the lower and upper bounds of the green screen color in YCbCr color space
lower_green_ycbcr = np.array([0, 135, 85])
upper_green_ycbcr = np.array([255, 180, 135])

# Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the green screen using HSV color space
    mask_hsv = cv2.inRange(hsv, lower_green_hsv, upper_green_hsv)

    # Convert the frame to YCbCr color space
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Create a mask for the green screen using YCbCr color space
    mask_ycbcr = cv2.inRange(ycbcr, lower_green_ycbcr, upper_green_ycbcr)

    # Combine the masks
    mask = cv2.bitwise_or(mask_hsv, mask_ycbcr)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to the frame and background
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(background, background, mask=mask)

    # Combine the foreground and background
    out = cv2.add(fg, bg)

    # Display the resulting frame
    cv2.imshow('Green Screen Removal', out)

    # Check for quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
```


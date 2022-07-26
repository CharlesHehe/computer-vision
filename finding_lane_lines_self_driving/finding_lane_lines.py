# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# reading in an image
image = cv2.imread('test_images/video_yellow_lane_left_12s.jpg')

print(type(image), image.shape)

imgHLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
white_lower = np.array([np.round(0 / 2), np.round(0.75 * 255), np.round(0.00 * 255)])
white_upper = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(0.30 * 255)])
white_mask = cv2.inRange(imgHLS, white_lower, white_upper)

yellow_lower = np.array([np.round(40 / 2), np.round(0.00 * 255), np.round(0.35 * 255)])
yellow_upper = np.array([np.round(60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
yellow_mask = cv2.inRange(imgHLS, yellow_lower, yellow_upper)

# plt.imshow(white_mask)
# plt.show()
#
# plt.imshow(yellow_mask)
# plt.show()
# Calculate combined mask, and masked image
mask = cv2.bitwise_or(yellow_mask, white_mask)
masked = cv2.bitwise_and(image, image, mask=mask)

# Write output images

# Grey
masked[:, :, 0] = 0
masked[:, :, 2] = 0
plt.imshow(masked)
plt.show()

blur = cv2.GaussianBlur(masked, (11, 11), 0)
plt.imshow(blur)
plt.show()

canny = cv2.Canny(blur, 50, 150)
plt.imshow(canny)
plt.show()

import cv2
import numpy as np

# Load the image
# img = cv2.imread("Images\\chessboard_imgs_sel_crop\\kFM1C.jpg")
img = cv2.imread("output\\calibration\\chessboard-images\\20221205_084959.jpg")
# img = cv2.imread("Images\\chessboard_imgs_sel_crop\\frame20.jpg")

chessboard_corners = (7, 10)

# Color-segmentation to get binary mask
lwr = np.array([0, 0, 143])
upr = np.array([179, 61, 252])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
msk = cv2.inRange(hsv, lwr, upr)

# Extract chess-board
krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
dlt = cv2.dilate(msk, krn, iterations=5)
res = 255 - cv2.bitwise_and(dlt, msk)

# Displaying chess-board features
res = np.uint8(res)
# cv2.imshow('res', res)
# cv2.waitKey(0)
ret, corners = cv2.findChessboardCorners(img, chessboard_corners,
                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
if ret:
    print(corners)
    fnl = cv2.drawChessboardCorners(img, chessboard_corners, corners, ret)
    cv2.imshow("fnl", fnl)
    cv2.waitKey(0)
else:
    print("No Checkerboard Found")
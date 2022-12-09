import cv2
import pathlib as pl
import numpy as np

# Load the image
# img = cv2.imread("Images\\chessboard_imgs_sel_crop\\kFM1C.jpg")
# img = cv2.imread("output\\calibration\\chessboard-images\\20221205_084959.jpg")
# img = cv2.imread("Images\\chessboard_imgs_sel_crop\\frame20.jpg")
p_base = pl.Path('C:\\Users\daniel.abler\CloudStorage\SWITCHdrive\WORK\Projects\ACTIVE\\2022_SENSE_InstrumentedSlippers\data\\2022-12-06_marker-tests')
p_video = p_base.joinpath('camera-videos\chessboard3d_video2.avi')
p_out = p_base.joinpath('chessboard-calibration-imgs')
p_out.mkdir(exist_ok=True, parents=True)
chessboard_corners = (7, 10)


video = cv2.VideoCapture(p_video.as_posix())

count_frame = 1
count_found = 1
while True:
    ret_video, img = video.read()

    if ret_video is False:
        break

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
        p_img = p_out.joinpath("frame%d.jpg" % count_found)
        cv2.imwrite(p_img.as_posix(), img)  # save frame as JPEG file
        print(f"- found {count_found} / {count_frame}")
        count_found += 1

    # if ret:
    #     print(corners)
    #     fnl = cv2.drawChessboardCorners(img, chessboard_corners, corners, ret)
    #     cv2.imshow("fnl", fnl)
    #     cv2.waitKey(0)
    #
    else:
        print("No Checkerboard Found")

    count_frame += 1

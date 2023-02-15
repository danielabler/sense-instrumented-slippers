import numpy as np
import cv2
import matplotlib.pylab as plt
import matplotlib as mpl

from analysis.calibration.config_calibration import page_size, page_height_pixel, page_width_pixel, aruco_dict, \
    n_columns, n_rows, square_size, marker_size, grid_size_width_pixel, grid_size_height_pixel, p_calibration_grid


page = 255*np.ones((page_height_pixel, page_width_pixel), dtype=np.uint8)

board = cv2.aruco.CharucoBoard_create(n_columns, n_rows, square_size, marker_size, aruco_dict)
imboard = board.draw((grid_size_width_pixel, grid_size_height_pixel))

width_start = 600
width_end   = width_start + imboard.shape[0]
height_start = 600
height_end   = height_start + imboard.shape[1]

page[width_start:width_end, height_start:height_end] = imboard

cv2.imwrite(p_calibration_grid.as_posix(), page)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(page, cmap = mpl.cm.gray, interpolation = "nearest")
ax.axis("off")
plt.show()

import numpy as np
import cv2, PIL, os
from cv2 import aruco
import pathlib as pl
import matplotlib.pylab as plt
import matplotlib as mpl

import pandas as pd

page_size = 'A4'
inch_to_mm = 25.4
resolution = 600 #dots/inch
square_size = 20
marker_size = 15
n_rows      = 11# -> height
n_columns   = 8 # -> width

grid_size_width  = square_size * n_columns
grid_size_width_inch = grid_size_width/inch_to_mm
grid_size_width_pixel= round(grid_size_width_inch * resolution)
grid_size_height = square_size * n_rows
grid_size_height_inch = round(grid_size_height/inch_to_mm)
grid_size_height_pixel = round(grid_size_height_inch * resolution)

page_sizes = {"A0": [840, 1188], "A1": [594, 840], "A2": [420, 594], "A3": [297, 420], "A4": [210, 297],
              "A5": [148, 210]}
page_width = page_sizes[page_size][0]
page_width_inch = page_width/inch_to_mm
page_width_pixel = round(page_width_inch * resolution)

page_height = page_sizes[page_size][1]
page_height_inch = page_height/inch_to_mm
page_height_pixel = round(page_height_inch * resolution)

page = 255*np.ones((page_height_pixel, page_width_pixel), dtype=np.uint8)


workdir = pl.Path("C:\\Users\daniel.abler\CloudStorage\SWITCHdrive\WORK\Projects\ACTIVE\\2022_SENSE_InstrumentedSlippers\data\\2022-12-06_marker-tests\charuco-calibration")
workdir.mkdir(exist_ok=True, parents=True)
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
board = aruco.CharucoBoard_create(n_columns, n_rows, square_size, marker_size, aruco_dict)
imboard = board.draw((grid_size_width_pixel, grid_size_height_pixel))

width_start = 600
width_end   = width_start + imboard.shape[0]
height_start = 600
height_end   = height_start + imboard.shape[1]


page[width_start:width_end, height_start:height_end] = imboard

cv2.imwrite(workdir.joinpath("chessboard.tiff").as_posix(), page)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(page, cmap = mpl.cm.gray, interpolation = "nearest")
ax.axis("off")
plt.show()
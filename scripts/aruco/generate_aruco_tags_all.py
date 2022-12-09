import numpy as np

from utils import ARUCO_DICT
import pathlib as pl
import cv2
import numpy as np
import sys

aruco_type = 'DICT_5X5_50'
aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

tag_size   = 200
p_base = pl.Path("C:\\Users\\daniel.abler\\Documents\\repositories\\instrumented-slippers\\output\\aruco_tags")
p_tags = p_base.joinpath('aruco_tags').joinpath(aruco_type)
p_tags.mkdir(exist_ok=True, parents=True)
tag_list = []
for aruco_id in range(50):
	print(f"-- ID '{aruco_id}'")
	tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
	cv2.aruco.drawMarker(aruco_dict, aruco_id, tag_size, tag, 1)

	# Save the tag generated
	tag_name = p_tags.joinpath(f"{aruco_type}_{aruco_id}.png").as_posix()
	cv2.imwrite(tag_name, tag)
	# cv2.imshow("aruco Tag", tag)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	tag_list.append(tag)



import skimage.io
import skimage.util

img_list = []
for aruco_id in range(50):
	tag_name = p_tags.joinpath(f"{aruco_type}_{aruco_id}.png").as_posix()
	img = skimage.io.imread(tag_name)
	img_list.append(img)

img_montage = skimage.util.montage(img_list, fill=255, grid_shape=(9, 6), padding_width=int(tag_size/5))
skimage.io.imsave(p_tags.joinpath("montage.png"), img_montage)

#
# from PIL import Image, ImageOps
#
# original_image = Image.open(p_tags.joinpath("montage.png").as_posix())
# width, height = original_image.size
#
#
# fit_and_resized_image = ImageOps.fit(original_image, size, Image.ANTIALIAS)
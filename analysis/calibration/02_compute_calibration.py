import pathlib as pl
import analysis.calibration.config_calibration as config
from modules.video_helpers import video_extract_frames_detected
import numpy as np
import cv2
from modules.calibration_helpers import get_corners_ids_from_imgs, compute_calibration

# p_video = config.p_raw_data.joinpath('2022-12-06_marker-tests').joinpath('chessboard3d_video2.avi')
p_video = config.p_raw_data.joinpath('2023-01-12').joinpath('08_calibration-pattern_video2.avi')

p_out   = config.p_calibration_results.joinpath('extracted-frames_calibration-grid')

#-- extract relevant images from video
video_extract_frames_detected(p_video, p_out, config.aruco_dict, save_imgs_markers=True, show=False, n_max=None, overwrite=False)

#-- compute calibration
p_img = pl.Path('/media/dabler/ext1TB/instrumented-slippers/processing-results/calibration/extracted-frames_calibration-grid/selected_imgs/08_calibration-pattern_video2_frame-125.png')

board = cv2.aruco.CharucoBoard_create(config.n_columns, config.n_rows, config.square_size, config.marker_size, config.aruco_dict)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

ids_found_list, corners_found_list, img_size = get_corners_ids_from_imgs(p_imgs=p_img.parent,
                                                                         aruco_dict=config.aruco_dict,
                                                                         board=board,
                                                                         criteria=criteria,
                                                                         glob_str='*.png')

ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors = \
                                    compute_calibration(corners_found_list, ids_found_list, img_size, board, criteria)

np.save(config.p_calibration_matrix, camera_matrix)
np.save(config.p_distortion_coefficients, distortion_coefficients0)

import pathlib as pl
import analysis.calibration.config_calibration as config
from modules.video_helpers import video_extract_frames_detected
import numpy as np
import cv2
import pickle
import logging
from modules.calibration_helpers import get_corners_ids_from_imgs, compute_calibration

# p_video = config.p_raw_data.joinpath('2022-12-06_marker-tests').joinpath('chessboard3d_video2.avi')
p_video = config.p_raw_data.joinpath('2023-01-12').joinpath('08_calibration-pattern_video2.avi')

p_out   = config.p_calibration_results.joinpath('extracted-frames_calibration-grid')

#-- extract relevant images from video
video_extract_frames_detected(p_video, p_out, config.aruco_dict, save_imgs_markers=True, show=False, n_max=None, overwrite=False)

#-- compute calibration
p_imgs = p_out.joinpath('selected_imgs')

board = cv2.aruco.CharucoBoard_create(config.n_columns, config.n_rows, config.square_size, config.marker_size, config.aruco_dict)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

ids_found_list, corners_found_list, img_size = get_corners_ids_from_imgs(p_imgs=p_imgs,
                                                                         aruco_dict=config.aruco_dict,
                                                                         board=board,
                                                                         criteria=criteria,
                                                                         glob_str='*.png')

error_max = 10
max_error_max = 1
iteration = 0
ids_to_process = ids_found_list.copy()
corners_to_process = corners_found_list.copy()

while error_max > max_error_max:
    logging.info(f"=== iteration {iteration} -- n={len(ids_to_process)}")
    
    ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = \
                                        compute_calibration(corners_to_process, ids_to_process, img_size, board, criteria)
    
    error_max = np.max(perViewErrors)
    indices_error_lt_max = list(np.where(perViewErrors < error_max*0.8)[0])
    ids_to_process = [ids_to_process[i] for i in indices_error_lt_max]
    corners_to_process = [corners_to_process[i] for i in indices_error_lt_max]

    results_dict = {'camera_matrix' : camera_matrix,
                    'distortion_coefficients' : distortion_coefficients0,
                    'rotation_vectors' : rotation_vectors,
                    'translation_vectors' : translation_vectors,
                    'std_intrinsic' : stdDeviationsIntrinsics,
                    'std_extrinsic' : stdDeviationsExtrinsics,
                    'reprojection_errors' : perViewErrors}
    p_calibration_results_dict = config.p_calibration_results.joinpath(f'calibration-results-dict_iter-{iteration}_error-max-{np.round(error_max,2)}.pkl')

    with open(p_calibration_results_dict, 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"f    - error_max = {error_max}")
    logging.info(f"f    - error_max < {max_error_max} -> {error_max<max_error_max}")
    logging.info(f"f    - new n     = {len(ids_to_process)}")


np.save(config.p_calibration_matrix, camera_matrix)
np.save(config.p_distortion_coefficients, distortion_coefficients0)

with open(config.p_calibration_results_dict, 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


import pathlib as pl
import analysis.calibration.config_calibration as config
from modules.video_helpers import video_extract_frames_detected


# p_video = config.p_raw_data.joinpath('2022-12-06_marker-tests').joinpath('chessboard3d_video2.avi')
p_video = config.p_raw_data.joinpath('2023-01-12').joinpath('08_calibration-pattern_video2.avi')

p_out   = config.p_calibration_results.joinpath('extracted-frames_calibration-grid')

video_extract_frames_detected(p_video, p_out, config.aruco_dict, save_imgs_markers=True, show=False, n_max=None, overwrite=True)

import cv2
from modules.aruco_helpers import aruco_display
import logging

def video_extract_frames_detected(p_video, output_dir, aruco_dict, save_imgs_markers=True, show=False, n_max=None,
                                  overwrite=False):

    name = p_video.name.split('.')[0]
    p_out_imgs_sel = output_dir.joinpath('selected_imgs')
    p_out_imgs_sel.mkdir(exist_ok=True, parents=True)
    p_out_imgs_sel_aruco = output_dir.joinpath('selected_imgs_markers')
    p_out_imgs_sel_aruco.mkdir(exist_ok=True, parents=True)

    if p_video.exists:
        if not p_out_imgs_sel.exists() or overwrite:
            logging.info(f'-- Extracting video {p_video} to {output_dir}')
            aruco_params = cv2.aruco.DetectorParameters_create()

            video = cv2.VideoCapture(p_video.as_posix())

            counter = 0
            counter_extr = 0
            while True:
                ret, frame_orig = video.read()
                counter = counter + 1
                logging.debug(f"   - reading frame {counter}")
                if ret is False:
                    break

                h, w, _ = frame_orig.shape

                width=1000
                height = int(width*(h/w))
                frame = cv2.resize(frame_orig, (width, height), interpolation=cv2.INTER_CUBIC)
                corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

                detected_markers = aruco_display(corners, ids, rejected, frame)
                if show:
                    cv2.imshow("Image", detected_markers)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        break

                if ids is not None:
                    logging.debug(f"            -> {len(ids)} markers detected")
                    p_out_imgs_sel_file         = p_out_imgs_sel.joinpath(f"{name}_frame-{counter}.png")
                    cv2.imwrite(p_out_imgs_sel_file.as_posix(), frame_orig)
                    if save_imgs_markers:
                        p_out_imgs_sel_aruco_file   = p_out_imgs_sel_aruco.joinpath(f"{name}_markers_frame-{counter}.png")
                        cv2.imwrite(p_out_imgs_sel_aruco_file.as_posix(), detected_markers)
                    counter_extr = counter_extr + 1
                else:
                    logging.debug(f"            -> no markers detected")

                if n_max is not None:
                    if n_max > counter:
                        break

            if show:
                cv2.destroyAllWindows()
            video.release()
            logging.info(f'-- Processed {counter} frames, {counter_extr} extracted')

        else:
            logging.info(f"-- Path {p_out_imgs_sel} already exists -- skipping")
    else:
        logging.fatal(f"-- Video file {p_video} does not exist")


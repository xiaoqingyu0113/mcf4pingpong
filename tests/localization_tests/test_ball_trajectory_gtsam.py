import cv2
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from mcf4pingpong.camera import triangulate
import mcf4pingpong.io as io
from mcf4pingpong import draw_util
from mcf4pingpong.estimator import IsamSolver
import glob
import mcf4pingpong.parameters as param
import time


def test_trajectory_triangulation():
    folder ='data/images/topspin'
    annotations = io.read_image_annotation(folder)

    camera_param_paths = ['config/camera/22276213_calibration.yaml', 
                          'config/camera/22276209_calibration.yaml',
                          'config/camera/22276216_calibration.yaml']
    
    camera_param_list = [io.read_camera_params(path) for path in camera_param_paths]
    isam_solver = IsamSolver(camera_param_list,
                    Cd = param.C_d,
                    Cm=param.C_m,
                    ez=param.ez,
                    mu=param.mu, 
                    ground_z0=0.030,
                    spin_prior = np.zeros(3), 
                    verbose = False)
    
    prev_annotes = None
    trajectory = []
    trajectory_isam = []
    time_elapsed = - time.time()

    for ann_idx, annotes in enumerate(annotations[3:]):
        iter = int(annotes['img_name'][5:11])
        t = float(annotes['time_in_seconds'])


        # if iter < 1793:
        #     continue
        # if iter > 2300:
        #     break
    

        # Yolo Detection has results
        if (prev_annotes is not None) and len(annotes['detections']) > 0:
            camera_id_left = int(prev_annotes['img_name'][3]) - 1
            camera_id_right = int(annotes['img_name'][3]) - 1
            # filter non pairs
            if camera_id_left == camera_id_right:
                continue


            #  pairwise localization:
            ball_position_candidates = []
            for detection_left in prev_annotes['detections']:
                for detection_right in annotes['detections']:
                    bbox_left = np.array(detection_left[2]) # detection = (name, prob, bbox)
                    bbox_right  = np.array(detection_right[2])
                    uv_left = bbox_left[:2] *np.array([1280/1024, 1024/768]) # resize to original
                    uv_right = bbox_right[:2] *np.array([1280/1024, 1024/768])
                    ball_position = triangulate(uv_left, uv_right, camera_param_list[camera_id_left], camera_param_list[camera_id_right])
                    # check backprop errors
                    uv_left_bp = camera_param_list[camera_id_left].proj2img(ball_position)
                    uv_right_bp = camera_param_list[camera_id_right].proj2img(ball_position)
                    bp_error = max([np.linalg.norm(uv_left - uv_left_bp), np.linalg.norm(uv_right - uv_right_bp)])
                    
                    if bp_error < 6.0:
                        ball_position_candidates.append((ball_position,uv_right))
                    # print(f"iter {iter}, bp_error = {bp_error}")
            # no pairs
            if len(ball_position_candidates) > 0:
        
                # with pairs, choose the best candidates
                launcher_pos = np.array([1.525/2 - 0.711/2, 0.711/2 - 2.74 , 0.2 ]) # launcher position

                # first in trajectory
                if len(trajectory) ==0:
                    referenced_position = launcher_pos

                    # choose best
                    best_dist = np.inf; best_pos = None; best_uv_right = None
                    for pos, uv_right in  ball_position_candidates:
                        pos_dis = np.linalg.norm(referenced_position - pos)
                        if pos_dis < best_dist:
                            best_dist = pos_dis
                            best_pos = pos
                            best_uv_right = uv_right
                else:
                    # check if new ball launched first
                    for pos, _ in  ball_position_candidates:
                        if (np.linalg.norm(launcher_pos - pos) < 0.5) and (np.linalg.norm(trajectory[-1] - pos) > 0.5):
                            print(f'iter {iter},new ball launched')
                            referenced_position = launcher_pos
                            isam_solver.reset()
                            break
                    else:
                        referenced_position = trajectory[-1]

                    # choose best
                    best_dist = np.inf; best_pos = None; best_uv_right = None
                    for pos, uv_right in  ball_position_candidates:
                        pos_dis = np.linalg.norm(referenced_position - pos)
                        if pos_dis < best_dist:
                            best_dist = pos_dis
                            best_pos = pos
                            best_uv_right = uv_right
                
                if best_pos[2] > -0.010 and best_dist < 0.5:
                    trajectory.append(best_pos)

                    ball_position_isam  = isam_solver.estimate([t, camera_id_right, best_uv_right[0], best_uv_right[1]], pos_prior=best_pos)
                    if ball_position_isam is not None:
                        trajectory_isam.append(ball_position_isam)


        if len(annotes['detections']) > 0:
            prev_annotes = annotes

    trajectory = np.array(trajectory)
    trajectory_isam = np.array(trajectory_isam)
    print(f'inference time for GTSAM = {len(trajectory_isam)/(time.time() +time_elapsed)} Hz')

    # print(trajectory)
    # print(trajectory_isam)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(trajectory[:,0],trajectory[:,1],trajectory[:,2],s=3, label='triangulation')
    ax.scatter(trajectory_isam[:,0],trajectory_isam[:,1],trajectory_isam[:,2],s=3, label='gtsam')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    for cm in camera_param_list:
        cm.draw(ax,scale=0.20)
    draw_util.set_axes_equal(ax)
    draw_util.set_axes_pane_white(ax)
    draw_util.draw_pinpong_table_outline(ax)

    plt.show()


if __name__ == '__main__':
    # test_projection()
    test_trajectory_triangulation()

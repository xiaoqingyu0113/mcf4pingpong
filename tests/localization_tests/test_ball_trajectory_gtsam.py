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


def test_trajectory_triangulation():
    folder ='data/images/nospin'
    annotations = io.read_image_annotation(folder)

    camera_param_paths = ['config/camera/22276213_calibration.yaml', 
                          'config/camera/22276209_calibration.yaml',
                          'config/camera/22276216_calibration.yaml']
    
    camera_param_list = [io.read_camera_params(path) for path in camera_param_paths]

    prev_annotes = None
    trajectory = []
    for annotes in annotations[3:]:
        iter = int(annotes['img_name'][5:11])
        
        if iter > 1200:
            break
        # if int(annotes['img_name'][3]) - 1 == 1:
        #     continue
        if (prev_annotes is not None) and len(annotes['detections']) > 0:
           
            camera_id_left = int(prev_annotes['img_name'][3]) - 1
            camera_id_right = int(annotes['img_name'][3]) - 1

            if camera_id_left == camera_id_right:
                continue
            # print(annotes['img_name'], '\t sec =', annotes['time_in_seconds'])
            detection_left = prev_annotes['detections'][0] # temporarily choose the first detection (name, prob, bbox)
            detection_right = annotes['detections'][0]

            bbox_left = np.array(detection_left[2])
            bbox_right  = np.array(detection_right[2])

            uv_left = bbox_left[:2] *np.array([1280/1024, 1024/768])
            uv_right = bbox_right[:2] *np.array([1280/1024, 1024/768])

            ball_position = triangulate(uv_left, uv_right, camera_param_list[camera_id_left], camera_param_list[camera_id_right])
            trajectory.append(ball_position)

        if len(annotes['detections']) > 0:
            prev_annotes = annotes

    trajectory = np.array(trajectory)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(trajectory[:,0],trajectory[:,1],trajectory[:,2],s=3)
    for cm in camera_param_list:
        cm.draw(ax,scale=0.20)
    draw_util.set_axes_equal(ax)
    draw_util.set_axes_pane_white(ax)
    draw_util.draw_pinpong_table_outline(ax)

    plt.show()


def test_trajectory_gtsam():
    folder ='data/images/nospin'
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
                    ground_z0=0.100,
                    spin_prior = np.zeros(3), 
                    verbose = True)

    prev_annotes = None
    trajectory = []
    trajectory_isam = []
    for annotes in annotations[3:]: # skip the first two frame
        frame_id = int(annotes['img_name'][5:11])
        t = float(annotes['time_in_seconds'])

        # filter
        if frame_id > 1200:
            break

        if (prev_annotes is not None) and len(annotes['detections']) > 0:
           
            camera_id_left = int(prev_annotes['img_name'][3]) - 1
            camera_id_right = int(annotes['img_name'][3]) - 1

            if camera_id_left == camera_id_right:
                continue
            # print(annotes['img_name'], '\t sec =', annotes['time_in_seconds'])
            detection_left = prev_annotes['detections'][0] # temporarily choose the first detection (name, prob, bbox)
            detection_right = annotes['detections'][0]

            bbox_left = np.array(detection_left[2])
            bbox_right  = np.array(detection_right[2])

            uv_left = bbox_left[:2] *np.array([1280/1024, 1024/768])
            uv_right = bbox_right[:2] *np.array([1280/1024, 1024/768])

            ball_position = triangulate(uv_left, uv_right, camera_param_list[camera_id_left], camera_param_list[camera_id_right])
            ball_position_isam  = isam_solver.estimate([t, camera_id_right, uv_right[0], uv_right[1]], pos_prior=ball_position)
            print(ball_position_isam)
            if ball_position_isam is not None:
                trajectory_isam.append(ball_position_isam)
            trajectory.append(ball_position)

        if len(annotes['detections']) > 0:
            prev_annotes = annotes

    trajectory = np.array(trajectory)
    trajectory_isam = np.array(trajectory_isam)
    print(trajectory_isam.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(trajectory[:,0],trajectory[:,1],trajectory[:,2],s=3, label='triangulation')
    ax.scatter(trajectory_isam[:,0],trajectory_isam[:,1],trajectory_isam[:,2],s=3, label='gtsam')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.legend()
    for cm in camera_param_list:
        cm.draw(ax,scale=0.20)
    draw_util.set_axes_equal(ax)
    draw_util.set_axes_pane_white(ax)
    draw_util.draw_pinpong_table_outline(ax)

    plt.show()
if __name__ == '__main__':
    # test_trajectory_triangulation()
    test_trajectory_gtsam()


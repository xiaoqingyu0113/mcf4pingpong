import cv2
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from mcf4pingpong.camera import triangulate
import mcf4pingpong.io as io
from mcf4pingpong import draw_util
from mcf4pingpong.estimator import IsamSolver, Estimator
import glob
import mcf4pingpong.parameters as param
import time


def test_trajectory_triangulation():
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
                    ground_z0=0.030,
                    spin_prior = np.zeros(3), 
                    verbose = False)
    
    estimator = Estimator(isam_solver, camera_param_list)

    prev_annotes = None
    trajectory = []
    trajectory_isam = []
    trajectory_spin = []
    trajectory_iter = []
    time_elapsed = - time.time()

    for annotes in annotations[3:]:
        rst = estimator.est(annotes)
        if rst[0] is not None:
            trajectory_isam.append(rst[0])
            trajectory.append(rst[1])
            trajectory_spin.append(estimator.isam_solver.get_w0())

    trajectory = np.array(trajectory)
    trajectory_isam = np.array(trajectory_isam)
    trajectory_spin = np.array(trajectory_spin)

    print(f'inference time for GTSAM = {len(trajectory_isam)/(time.time() +time_elapsed)} Hz')


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

  

    plt.show()

if __name__ == '__main__':
    test_trajectory_triangulation()

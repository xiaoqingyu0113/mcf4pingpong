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

def separate_trajectory(trajectory, trajectory_iter):
    '''
    the start and end iters for each individual trajectory
    '''
    iter_separates = []
    idx_separates = []
    start_idx = 0
    N =  len(trajectory)
    for idx in range(1, len(trajectory)):
        p_curr = trajectory[idx]
        p_prev = trajectory[idx-1]
        if np.linalg.norm(p_curr - p_prev) > 2.0:
            iter_separates.append([trajectory_iter[start_idx], trajectory_iter[idx - 1]])
            idx_separates.append([start_idx, idx - 1])
            start_idx = idx
    iter_separates.append([trajectory_iter[start_idx], trajectory_iter[N-1]])
    idx_separates.append([start_idx, N-1])
    return iter_separates, idx_separates


def test_trajectory_triangulation():
    folder ='data/images/ball_traj_collector_1'
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

    trajectory = []
    trajectory_isam = []
    # trajectory_spin = []
    trajectory_iter = []
    trajectory_time = []

    for annotes in annotations[3:]:
        iter = int(annotes['img_name'][5:11])
        rst = estimator.est(annotes)
        if rst[0] is not None:
            trajectory_isam.append(rst[0])
            trajectory.append(rst[1])
            trajectory_iter.append(iter)
            trajectory_time.append(float(annotes['time_in_seconds']))


    trajectory = np.array(trajectory)
    trajectory_isam = np.array(trajectory_isam)
    trajectory_time = np.array(trajectory_time)[:,None]

    trajectory_timestamped = np.hstack((trajectory_time, trajectory_isam))
    print(trajectory_timestamped.shape)
    # trajectory_spin = np.array(trajectory_spin)

    iter_separates, idx_separates = separate_trajectory(trajectory, trajectory_iter)
    print(len(idx_separates))
    print(idx_separates)

    save_folder = folder.replace('images','trajectories')
    i =0
    for sp in idx_separates:
        if sp[1] - sp[0] < 100:
            continue
        ti = list(range(sp[0],sp[1]-1))
        np.savetxt(save_folder+f'/{i}.csv', trajectory_timestamped[ti,:], delimiter=',')
        i+=1

    #     ti = list(range(sp[0],sp[1]-1))
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for sp in idx_separates:
    #     ti = list(range(sp[0],sp[1]-1))
    #     # ax.scatter(trajectory[:,0],trajectory[:,1],trajectory[:,2],s=3, label='triangulation')
    #     ax.scatter(trajectory_isam[ti,0],trajectory_isam[ti,1],trajectory_isam[ti,2],s=3, label='gtsam')
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_zlabel('Z (m)')
    # ax.legend()
    # for cm in camera_param_list:
    #     cm.draw(ax,scale=0.20)
    # draw_util.set_axes_equal(ax)
    # draw_util.set_axes_pane_white(ax)
    # draw_util.draw_pinpong_table_outline(ax)

    # plt.show()

 
if __name__ == '__main__':
    test_trajectory_triangulation()
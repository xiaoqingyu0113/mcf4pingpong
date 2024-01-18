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

def separate_trajectory(trajectory, trajectory_iter):
    '''
    the start and end iters for each individual trajectory
    '''
    separates = []
    start_idx = 0
    N =  len(trajectory)
    for idx in range(1, len(trajectory)):
        p_curr = trajectory[idx]
        p_prev = trajectory[idx-1]
        if np.linalg.norm(p_curr - p_prev) > 2.0:
            separates.append([trajectory_iter[start_idx], trajectory_iter[idx - 1]])
            start_idx = idx
    separates.append([trajectory_iter[start_idx], trajectory_iter[N-1]])
    return separates




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
    
    estimator = Estimator(isam_solver, camera_param_list)

    trajectory = []
    trajectory_isam = []
    trajectory_spin = []
    trajectory_iter = []

    for annotes in annotations[3:]:
        iter = int(annotes['img_name'][5:11])
        rst = estimator.est(annotes)
        if rst[0] is not None:
            trajectory_isam.append(rst[0])
            trajectory.append(rst[1])
            trajectory_iter.append(iter)


    trajectory = np.array(trajectory)
    trajectory_isam = np.array(trajectory_isam)
    trajectory_spin = np.array(trajectory_spin)

    separates = separate_trajectory(trajectory, trajectory_iter)
    print(separates)


    # start labeling
    spin_prior_guess = np.random.rand(3)
    guess_error = np.inf # relative percentage
    trajectory_w0 = []
    bounce_idx = 0
    while guess_error > 0.0017:
        estimator.reset()
        estimator.isam_solver.spin_prior = spin_prior_guess
        start_iter,  end_iter = separates[0] # use the chosen trajectory to try
        trajectory_w0 = []
        for annotes in annotations[3:]:
            iter = int(annotes['img_name'][5:11])
            if iter < start_iter:
                continue
            if iter > end_iter:
                break
            rst = estimator.est(annotes)
            if rst[0] is not None:
                w0 = estimator.isam_solver.get_w0()
                trajectory_w0.append(w0)
                if estimator.isam_solver.spin_idx >0:
                    bounce_idx = len(trajectory_w0)



        print(f'new w0 = {w0}')
        guess_error  = np.linalg.norm(w0 - spin_prior_guess)/np.linalg.norm(w0)
        print(f'guess error = {guess_error}')
        spin_prior_guess = w0
        print(f'bounce_idx = {bounce_idx}')
        
    
    trajectory_w0 = np.array(trajectory_w0)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(np.arange(len(trajectory_w0)),trajectory_w0[:,0],label='w0_x')
    plt.show()

if __name__ == '__main__':
    test_trajectory_triangulation()

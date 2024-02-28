import cv2
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from mcf4pingpong.camera import triangulate
import mcf4pingpong.io as io
from mcf4pingpong import draw_util
from mcf4pingpong.estimator import IsamSolver, Estimator
from mcf4pingpong.predictor import predict_trajectory
import glob
import mcf4pingpong.parameters as param
import time


def test_prediction():
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
                    ground_z0=0.035,
                    spin_prior = np.zeros(3), 
                    verbose = False)
    
    estimator = Estimator(isam_solver, camera_param_list)
    tspan = np.linspace(0,1.0,100)

    view_in_camera = 3
    cam_param = camera_param_list[view_in_camera-1]
    for annotes in annotations[3:]:
        rst = estimator.est(annotes)
        img_name = annotes['img_name']
        img_name = f'{folder}/{img_name}' # relative path

        if (rst[0] is not None) and (f'cam{view_in_camera}' in img_name):
            img = cv2.imread(img_name)
            H,W,C = img.shape
            xN = predict_trajectory(rst[0][:3], 
                               rst[0][3:6], 
                               rst[0][6:9], 
                               tspan, 
                               C_d=param.C_d, 
                               C_m=param.C_m, 
                               mu = param.mu,
                                ez = param.ez)
            # print(xN[:,:3])
            u,v = cam_param.proj2img(xN[:,:3],shape=(H,W)).T
            valid_indices = (u>=0) & (v>=0) & (u<W) & (v<H)
            u = u[valid_indices].astype(int)
            v = v[valid_indices].astype(int)
            points = np.vstack((u, v)).T.reshape(-1, 1, 2)
            img = cv2.polylines(img, [points], False, (0,0,255), 2)
            cv2.imwrite(img_name.replace('images','debug_prediction'), img)
            
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.plot(xN[:,0],xN[:,1],xN[:,2])
            # ax.set_xlabel('X (m)')
            # ax.set_ylabel('Y (m)')
            # ax.set_zlabel('Z (m)')
            # for cm in camera_param_list:
            #     cm.draw(ax,scale=0.20)
            # draw_util.set_axes_equal(ax)
            # draw_util.set_axes_pane_white(ax)
            # draw_util.draw_pinpong_table_outline(ax)
            # fig.savefig(img_name.replace('images','debug_prediction').replace('jpg','png'))
            
    



    


if __name__ == '__main__':
    test_prediction()

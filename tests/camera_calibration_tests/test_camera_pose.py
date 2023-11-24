import cv2
from cv2 import aruco
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

from mcf4pingpong.io import read_yaml_file, write_yaml_file, read_camera_params
from mcf4pingpong import camera, draw_util
import glob
# opencv 4.8.0, new api




def rvec2rotm(rvec):
    '''
    Rodrigues: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
    '''
    th = np.linalg.norm(rvec)
    rvec = rvec / th
    rotm = np.cos(th)*np.eye(3) + (1-np.cos(th))*(rvec[:,None] @ rvec[None,:]) + np.sin(th)*np.array([[0, -rvec[2], rvec[1]],[rvec[2], 0, -rvec[0]],[-rvec[1], rvec[0], 0]])
    return rotm


def rot_x(angle_in_degrees):
    # Define the rotation matrix for 180-degree rotation about the x-axis
    R = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(angle_in_degrees)), -np.sin(np.radians(angle_in_degrees))],
        [0, np.sin(np.radians(angle_in_degrees)), np.cos(np.radians(angle_in_degrees))]
    ])
    
    # Rotate the vector
    return R

def estimate_aruco_pose(corners, marker_size,camera_matrix,dist_coeffs):
    corners = corners[0]
    object_points = np.array([
        [-marker_size/2, -marker_size/2, 0],
        [-marker_size/2, marker_size/2, 0],
        [marker_size/2, marker_size/2, 0],
        [marker_size/2, -marker_size/2, 0]
    ])
    _, rvecs, tvecs = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    return rvecs, tvecs


def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected_img_points = detector.detectMarkers(frame)

    if len(corners) > 0:
        marker_size = 0.600
        
        rvecs, tvecs = estimate_aruco_pose(corners,marker_size,matrix_coefficients, distortion_coefficients)

        rvecs = rvecs.ravel()
        tvecs = tvecs.ravel()

        # Draw Axis
        cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvecs, tvecs, 0.6)
    else:
        raise

    return frame,rvec2rotm(rvecs),tvecs

def test_camera_pose_estimation():
    '''
    Assume the intrinsics has been calibrated, and saved in config/camera/$(SERIAL_#)_calibration.yaml
    Extrinsics will also be saved in the same file
    '''

    april_paths = glob.glob('data/april_tag/cam*')
    april_paths = sorted(april_paths, key= lambda x: int(x[18]))
    # print(april_paths)
    # april_paths = ['data/april_tag/cam1_000002.jpg', 'data/april_tag/cam2_000000.jpg','data/april_tag/cam3_000001.jpg' ]
    camera_serials = ['22276213', '22276209', '22276216']


    aruco_dict_type = cv2.aruco.DICT_APRILTAG_36h11

    for april_path, cam_serial in zip(april_paths,camera_serials):
        image = cv2.imread(april_path)
        yaml_path = f'config/camera/{cam_serial}_calibration.yaml'
        cam_param = read_yaml_file(yaml_path)
        k = np.array(cam_param['camera_matrix']['data']).reshape(3,3)
        d = np.array(cam_param['distortion_coefficients']['data'])

        annotated_image, R_w_c, t_w_c = pose_esitmation(image, aruco_dict_type, k, d)
        
        cam_param['rotation_matrix'] = {'rows':3, 'cols':3, 'data': R_w_c.reshape(-1).tolist()}
        cam_param['translation'] = t_w_c.tolist()

        cv2.imwrite(april_path.replace('cam','debug_cam'),annotated_image)
        write_yaml_file(yaml_path, cam_param)


def draw_camera_poses():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    camera_param_paths = ['config/camera/22276213_calibration.yaml', 
                          'config/camera/22276209_calibration.yaml',
                          'config/camera/22276216_calibration.yaml']
    
    camera_param_list = [read_camera_params(path) for path in camera_param_paths]

    for cp in camera_param_list:
        cp.draw(ax,scale=0.3)
    draw_util.draw_pinpong_table_outline(ax)
    draw_util.set_axes_equal(ax)
    plt.show()



if __name__ == '__main__':
    # write pose
    # test_camera_pose_estimation()
    # show cameras
    draw_camera_poses()
    # debug()
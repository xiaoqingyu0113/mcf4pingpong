import cv2
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from mcf4pingpong.camera import triangulate
import mcf4pingpong.io as io
from mcf4pingpong import draw_util


def test_projection():
    cam_param = io.read_camera_params('config/camera/22276213_calibration.yaml')
    point3d = np.array([0,0,0,1])
    uv1 = M = cam_param.get_projection_matrix() @ point3d
    uv = uv1[:2]/uv1[2]; uv=uv.astype('int')
    print(uv)

    image = cv2.imread('data/april_tag/cam1_000002.jpg')
    cv2.circle(image, (uv[0],uv[1]), 6, (0,0,255), 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.show()

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
        if int(annotes['img_name'][3]) - 1 == 1:
            continue
        if (prev_annotes is not None) and len(annotes['detections']) > 0:
           
            camera_id_left = int(prev_annotes['img_name'][3]) - 1
            camera_id_right = int(annotes['img_name'][3]) - 1

            if camera_id_left == camera_id_right:
                continue
            print(annotes['img_name'], '\t sec =', annotes['time_in_seconds'])
            detection_left = prev_annotes['detections'][0] # temporarily choose the first detection (name, prob, bbox)
            detection_right = annotes['detections'][0]

            bbox_left = np.array(detection_left[2])
            bbox_right  = np.array(detection_right[2])

            uv_left = bbox_left[:2] + bbox_left[2:]/2
            uv_right = bbox_right[:2] + bbox_right[2:]/2

            # uv_left = camera_param_list[camera_id_left].undistort_pixel(uv_left)
            # uv_right = camera_param_list[camera_id_right].undistort_pixel(uv_right)

            ball_position = triangulate(uv_left, uv_right, camera_param_list[camera_id_left], camera_param_list[camera_id_right])
            trajectory.append(ball_position)

        if len(annotes['detections']) > 0:
            prev_annotes = annotes

    trajectory = np.array(trajectory)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(trajectory[:,0],trajectory[:,1],trajectory[:,2])
    for cm in camera_param_list:
        cm.draw(ax,scale=0.10)
    draw_util.set_axes_equal(ax)
    draw_util.draw_pinpong_table_outline(ax)
    plt.show()

def test_manually_clicks():
    # im1 = plt.imread('data/april_tag/debug_cam3_000002.jpg')
    # plt.imshow(im1)
    # plt.show()

    uv_cam1 = np.array([[313,397],[184,397],[508,299]]).astype(float)
    uv_cam2 = np.array([[934,288],[927,241],[1254,334]]).astype(float)
    uv_cam3 = np.array([[852,433],[1004,400],[730,660]]).astype(float)

    camera_param_paths = ['config/camera/22276213_calibration.yaml', 
                          'config/camera/22276209_calibration.yaml',
                          'config/camera/22276216_calibration.yaml']
    
    camera_param_list = [io.read_camera_params(path) for path in camera_param_paths]


    ball_position12 = triangulate(uv_cam1, uv_cam2, camera_param_list[0], camera_param_list[1]).reshape(-1,3).T
    ball_position23 = triangulate(uv_cam2, uv_cam3, camera_param_list[1], camera_param_list[2]).reshape(-1,3).T
    ball_position13 = triangulate(uv_cam1, uv_cam3, camera_param_list[0], camera_param_list[2]).reshape(-1,3).T
    

    # print(ball_position12)
    # print(ball_position23)
    # print(ball_position13)
    # uv1 = camera_param_list[0].proj2img(ball_position12)
    # print(uv1)


if __name__ == '__main__':
    # test_projection()
    test_trajectory_triangulation()
    # test_manually_clicks()

import cv2
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from mcf4pingpong.camera import triangulate
import mcf4pingpong.io as io
from mcf4pingpong import draw_util
import glob

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

def test_ball_center():
    img_path = 'data/images/nospin/'
    images = glob.glob(img_path + '*.jpg')
    images.sort()

    
    for img in images:
        im_array = cv2.imread(img)
        annote_path = img.replace('jpg','json')
        annote = io.read_json_file(img.replace('jpg','json'))
        if len(annote['detections']) >0:
            detection = annote['detections'][0]
            bbox = np.array(detection[2])
            uv = bbox[:2] 
            uv[0] = uv[0]/1024*1280
            uv[1] = uv[1]/768*1024
            uv = uv.astype(int)
            cv2.circle(im_array, (uv[0], uv[1]), 6, (255, 0, 0), -1)

        cv2.imshow('image', im_array)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # test_projection()
    test_trajectory_triangulation()
    # test_ball_center()

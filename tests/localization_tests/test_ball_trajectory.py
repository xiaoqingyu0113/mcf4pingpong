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
    folder ='data/images/sidespin'
    annotations = io.read_image_annotation(folder)

    camera_param_paths = ['config/camera/22276213_calibration.yaml', 
                          'config/camera/22276209_calibration.yaml',
                          'config/camera/22276216_calibration.yaml']
    
    camera_param_list = [io.read_camera_params(path) for path in camera_param_paths]

    prev_annotes = None
    trajectory = []
    for annotes in annotations[3:]:
        iter = int(annotes['img_name'][5:11])
        
        # if iter > 1200:
        #     break
        # if int(annotes['img_name'][3]) - 1 == 1:
        #     continue
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
                    
                    if bp_error < 10.0:
                        ball_position_candidates.append(ball_position)
                    # print(f"iter {iter}, bp_error = {bp_error}")
            # no pairs
            if len(ball_position_candidates) ==0:
                print(f"iter {iter}, no pairs")
            else:
                # with pairs, choose the best candidates
                launcher_pos = np.array([1.525/2 - 0.711/2, 0.711/2 - 2.74 , 0.2 ]) # launcher position

                if len(trajectory) ==0:
                    referenced_position = launcher_pos

                    # choose best
                    best_dist = np.inf; best_pos = None
                    for pos in  ball_position_candidates:
                        pos_dis = np.linalg.norm(referenced_position - pos)
                        if pos_dis < best_dist:
                            best_dist = pos_dis
                            best_pos = pos
                else:
                    # check if new ball launched first
                    for pos in  ball_position_candidates:
                        if (np.linalg.norm(launcher_pos - pos) < 0.2) and (np.linalg.norm(trajectory[-1] - pos) > 0.2):
                            print(f'iter {iter},new ball launched')
                            referenced_position = launcher_pos
                            break
                    else:
                        referenced_position = trajectory[-1]

                    # choose best
                    best_dist = np.inf; best_pos = None
                    for pos in  ball_position_candidates:
                        pos_dis = np.linalg.norm(referenced_position - pos)
                        if pos_dis < best_dist:
                            best_dist = pos_dis
                            best_pos = pos
                
                if best_pos[2] > -0.010 and best_dist < 0.2:
                    trajectory.append(best_pos)


        if len(annotes['detections']) > 0:
            prev_annotes = annotes

    trajectory = np.array(trajectory)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(trajectory[:,0],trajectory[:,1],trajectory[:,2],s=3)
    for cm in camera_param_list:
        cm.draw(ax,scale=0.20)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    draw_util.set_axes_equal(ax)
    draw_util.set_axes_pane_white(ax)
    draw_util.draw_pinpong_table_outline(ax)

    plt.show()


if __name__ == '__main__':
    # test_projection()
    test_trajectory_triangulation()

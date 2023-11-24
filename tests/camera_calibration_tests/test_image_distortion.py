import cv2
import numpy as np
from mcf4pingpong import camera, io
import matplotlib.pyplot as plt




def test_undistortion_vs_distortion():
    camera_param = io.read_camera_params('config/camera/22276209_calibration.yaml')
    x_coords, y_coords = np.meshgrid(np.arange(0,1280,50), np.arange(0,1080,50))
    pixel_coords = np.dstack((y_coords, x_coords)).astype(float)
    pixel_coords_undistort = pixel_coords.copy()

    height, width = x_coords.shape
    
    for w in range(width):
        for h in range(height):
            homogeneous_2d_points = cv2.undistortPoints(pixel_coords[h,w,::-1], camera_param.K, camera_param.d, None, camera_param.K)
            pixel_coords_undistort[h,w,:] = homogeneous_2d_points.flatten()
    

    compact_coords = pixel_coords_undistort.reshape(-1,2)

    x_un= compact_coords[:,0]
    y_un = compact_coords[:,1]

    compact_coords = pixel_coords.reshape(-1,2)
    x = compact_coords[:,1]
    y = compact_coords[:,0]

    plt.scatter(x,y,10,label='distorted')
    plt.scatter(x_un,y_un,10,label='undistorted')
    plt.quiver(x_coords, y_coords, pixel_coords_undistort[:,:,0] - pixel_coords[:,:,1], pixel_coords_undistort[:,:,1] - pixel_coords[:,:,0], angles='xy', scale_units='xy', scale=1,color='g')
    plt.title('raw image vs undistort image')
    plt.legend()
    plt.show()


def test_implemented_distortion():
    print('-----test: compare distort and undistort--------------')
    camera_param = io.read_camera_params('config/camera/22276209_calibration.yaml')
    uv = np.array([[0,0],[0,1000],[1000,0],[1000,1000]])

    uv_dist = camera_param.distort_pixel(uv)
    uv_undist = camera_param.undistort_pixel(uv_dist)


    print(f'original = \n{uv}\nundistort = \n{uv_undist}')

def test_compare_undistort_with_cv2():
    print('-----test: compare undistort layer with opencv--------------')
    camera_param = io.read_camera_params('config/camera/22276209_calibration.yaml')
    uv = np.array([[0,0],[0,1000],[1000,0],[1000,1000]])

    uv_dist = camera_param.distort_pixel(uv)

    # for p in uv:
    homogeneous_2d_points = cv2.undistortPoints(uv_dist, camera_param.K, camera_param.d, None, camera_param.K)
    
    print(homogeneous_2d_points)
    print(camera_param.undistort_pixel(uv_dist))
if __name__ == '__main__':
    test_undistortion_vs_distortion()
    test_implemented_distortion()
    test_compare_undistort_with_cv2()
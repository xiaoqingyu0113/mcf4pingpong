import cv2
import numpy as np
import mcf4pingpong.io as io
from mcf4pingpong.camera import triangulate

class ImageProcessor:
    def __init__(self):
        self.clicked_points = []

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
            # print((x, y))
            cv2.circle(param, (x, y), 3, (255, 0, 0), -1)
            cv2.imshow('image', param)

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', self.click_event, img)
        cv2.waitKey(0)

    def get_clicked_points(self):
        return self.clicked_points

def test_two_points_distance():
    '''
    choose two points from each cameras, and triangulate
    '''
    camera_param_paths = ['config/camera/22276213_calibration.yaml', 
                          'config/camera/22276209_calibration.yaml',
                          'config/camera/22276216_calibration.yaml']
    
    camera_param_list = [io.read_camera_params(path) for path in camera_param_paths]

    processor = ImageProcessor()
    image_paths=['data/april_tag/cam1_000000.jpg','data/april_tag/cam2_000008.jpg','data/april_tag/cam3_000002.jpg']

    # Loop through each image
    for path in image_paths[:-1]:
        processor.process_image(path)

    uvs = np.array(processor.get_clicked_points(),dtype=float)
    N = len(uvs)//2
    locs_3d = []
    for i in range(N):
        loc_3d = triangulate(uvs[i], uvs[i+N], camera_param_list[0], camera_param_list[1])
        locs_3d.append(loc_3d)
        print(loc_3d)

    print(np.linalg.norm(locs_3d[0] - locs_3d[1]))


    cv2.destroyAllWindows()



if __name__ == '__main__':
    test_two_points_distance()
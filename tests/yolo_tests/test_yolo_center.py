import cv2
from mcf4pingpong import io
import numpy as np
import glob



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
    test_ball_center()
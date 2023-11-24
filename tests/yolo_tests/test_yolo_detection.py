import cv2
import glob
import os
from tqdm import tqdm
import json
from mcf4pingpong.darknet import darknet

#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7#

'''
This test also compute the detections and write in json
'''

class YoloDetector:
    def __init__(self, config_file, data_file, weights):
        network, class_names, class_colors = darknet.load_network(config_file,data_file,weights)
        self.network = network
        self.class_names = class_names
        self.class_colors = class_colors
        self.thresh = 0.5
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(self.width, self.height, 3)
        self.prev_image= None
        
    def set_thresh(self, t):
        self.thresh = t
    
    def detect(self, image_path):
        image = cv2.imread(image_path)
        image_original = cv2.resize(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB), (self.width, self.height),
                                interpolation=cv2.INTER_LINEAR)
        
        image = self.background_subtract(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height),
                                interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=self.thresh)
        image = cv2.cvtColor(darknet.draw_boxes(detections, image_original, self.class_colors), cv2.COLOR_BGR2RGB)
        return detections, image
    
    def background_subtract(self,image, diff_thresh = 25):
        if self.prev_image is None:
            self.prev_image = image
            return image
        gray1 = cv2.cvtColor(self.prev_image, cv2.COLOR_BGR2GRAY).astype(int)
        gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(int)
        frame_diff = ((gray1 - gray2) < diff_thresh) & ((gray2 - gray1) < diff_thresh) 
     
        self.prev_image = image.copy()
        image[frame_diff,:] = 0
        return image


    def free(self):
        self.darknet.free_image(self.darknet_image)
    
    def clear_prev_image(self):
        self.prev_image = None

def separate_image_with_camera(jpg_files):
    cam1_jpg, cam2_jpg, cam3_jpg = [], [], []
    for jpg in jpg_files:
        if 'cam1' in jpg:
            cam1_jpg.append(jpg)
        elif 'cam2' in jpg:
            cam2_jpg.append(jpg)
        elif 'cam3' in jpg:
            cam3_jpg.append(jpg)
    cam1_jpg.sort()
    cam2_jpg.sort()
    cam3_jpg.sort()
    return cam1_jpg, cam2_jpg, cam3_jpg

def load_json_to_dict(filename):
    """
    Load a JSON file and return its contents as a Python dictionary.

    Parameters:
    - filename (str): The name of the JSON file to read.

    Returns:
    - dict: The loaded JSON data as a Python dictionary.
    """
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{filename}'.")
        return None
    
def save_dict_to_json(filename, data):
    """
    Save a Python dictionary to a JSON file.

    Parameters:
    - data (dict): The dictionary to be saved.
    - filename (str): The name of the JSON file to save.

    Returns:
    - None
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def get_default_params():
    config_file = 'config/darknet/yolov4-lite.cfg'
    data_file = 'config/darknet/obj.data'
    weights = 'config/darknet/yolov4-lite_final.weights'
    return config_file, data_file, weights

def test_single_image():
    config_file, data_file, weights = get_default_params()
    yolo_detector = YoloDetector(config_file, data_file, weights)
    detections,image = yolo_detector.detect('data/images/mix_spin/cam3_003877.jpg')
    print(detections)
    cv2.imwrite('output_image.jpg', image)

def test_all_images():
    '''
    This is a test of yolo detector for data/images/**/*.jpg
    The annotated images are saved in data/debug/**/*.jpg

    '''
    config_file, data_file, weights = get_default_params()
    yolo_detector = YoloDetector(config_file, data_file, weights)
    directories = glob.glob('data/images/*')
    for directory in tqdm(directories,desc='directories', leave=False):
        jpg_files = glob.glob(f'{directory}/*.jpg')

        # separate the images
        cam1_jpg, cam2_jpg, cam3_jpg = separate_image_with_camera(jpg_files)

        for jpg in tqdm(cam1_jpg,desc='process cam1'):
            detections, image = yolo_detector.detect(jpg)
            debug_directory = directory.replace('images','debug')
            if not os.path.exists(debug_directory):
                os.makedirs(debug_directory,exist_ok=True)

            # save images
            cv2.imwrite(str(jpg).replace('images','debug'), image)

            # save detections in json
            json_path = str(jpg).replace('jpg','json')
            dict_data = load_json_to_dict(json_path)
            dict_data['detections'] = detections
            save_dict_to_json(json_path, dict_data)
            
        yolo_detector.clear_prev_image()

        for jpg in tqdm(cam2_jpg,desc='process cam2',leave=False):
            detections, image = yolo_detector.detect(jpg)
            debug_directory = directory.replace('images','debug')
            if not os.path.exists(debug_directory):
                os.makedirs(debug_directory,exist_ok=True)

            # save images
            cv2.imwrite(str(jpg).replace('images','debug'), image)

            # save detections in json
            json_path = str(jpg).replace('jpg','json')
            dict_data = load_json_to_dict(json_path)
            dict_data['detections'] = detections
            save_dict_to_json(json_path, dict_data)
            
        yolo_detector.clear_prev_image()

        for jpg in tqdm(cam3_jpg,desc='process cam3'):
            detections, image = yolo_detector.detect(jpg)
            debug_directory = directory.replace('images','debug')
            if not os.path.exists(debug_directory):
                os.mkdir(debug_directory)

            # save images
            cv2.imwrite(str(jpg).replace('images','debug'), image)

            # save detections in json
            json_path = str(jpg).replace('jpg','json')
            dict_data = load_json_to_dict(json_path)
            dict_data['detections'] = detections
            save_dict_to_json(json_path, dict_data)
            
        yolo_detector.clear_prev_image()






if __name__ == '__main__':
    test_all_images()
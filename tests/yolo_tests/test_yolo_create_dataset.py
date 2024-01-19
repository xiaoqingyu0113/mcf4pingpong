import os
import shutil
import glob
import json

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

def write_txt_yolomark(json_file):
    d = load_json_to_dict(json_file)
    detections = d['detections']

    txt_file = json_file.replace('json','txt')

    with open(txt_file,'w') as file:
        for detect in detections:
            x,y,w,h = detect[2]
            str_list = ['0', str(x/1024), str(y/768),str(w/1024),str(h/768)]
            numbers_text = ' '.join(str_list)
            file.write(numbers_text+'\n')


    
def main():
    # shutil.copytree('data/images', '/home/core-robotics/yolo_training_tennis')
    # shutil.copytree('config/darknet', '/home/core-robotics/yolo_training_tennis/config')
    directories = glob.glob('/home/core-robotics/yolo_training_tennis/images/*')
    for directory in directories:
        json_files = glob.glob(f'{directory}/*.json')
        for json_file in json_files:
            write_txt_yolomark(json_file)



if __name__ == '__main__':
    main()
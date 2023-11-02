import yaml
import json
import numpy as np
from mcf4pingpong import camera
import glob


def read_image_annotation(folder_name):
    files = glob.glob(folder_name + '/*.json')
    annotations = []
    for f in files:
        data = read_json_file(f)
        annotations.append(data)
    annotations = sorted(annotations, key=lambda x: x['time_in_seconds'])
    return annotations

def read_camera_params(file_path):
    camera_param_raw = read_yaml_file(file_path)
    K  = np.array(camera_param_raw['camera_matrix']['data']).reshape(3,3)
    R =  np.array(camera_param_raw['rotation_matrix']['data']).reshape(3,3)
    t = np.array(camera_param_raw['translation'])
    d = np.array(camera_param_raw['distortion_coefficients']['data'])
    camera_param = camera.CameraParam(K,R,t,distortion=d)
    return camera_param

def read_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Load the YAML data into a Python dictionary
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the YAML file: {e}")
        return None
    
def write_yaml_file( file_path, data):
    try:
        with open(file_path, 'w') as file:
            # Dump the Python data to the YAML file
            yaml.dump(data, file)
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to the YAML file: {e}")


def write_json_file(filename, data):
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

def read_json_file(filename):
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
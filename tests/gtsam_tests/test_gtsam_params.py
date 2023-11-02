from mcf4pingpong import camera, io
import numpy as np

def test():
    camera_param = io.read_camera_params('config/camera/22276209_calibration.yaml')
    camera_param.test()
    

if __name__ == '__main__':
    test()




import mcf4pingpong.io as io
import numpy as np


camera_param = io.read_camera_params('config/camera/22276209_calibration.yaml')

p = np.array([0,0,0])

uv = camera_param.proj2img(p)


print(uv)
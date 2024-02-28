from mcf4pingpong.dynamics.dynamics import *
import numpy as np
import time
from jaxsam.util import timeit

for _ in range(100):
    with timeit():
        x = bounce_slide_velocity_forward(np.random.rand(3), np.random.rand(3), [0.1,0.2])

import os

import numpy

test = numpy.load(os.path.join(os.path.dirname(__file__), 'leap_hand_in_palm_cube_grasp_50k_s105.npy'))
print(test)
print(test[[2]])

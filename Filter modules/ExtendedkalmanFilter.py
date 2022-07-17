# Import libraries from python

import numpy as np
import filterpy.common as fpc
from filterpy.kalman import ExtendedKalmanFilter as EKFilter

# Number of state variables for the dim_x input extended kalman filter. e.g. if you are locating the position of a node in two dimensions, the number would be 3.

A = 3

# Number of measurement inputs for the dim_y input extended kalman filter. e.g. if the node is providing you with it's position, the number would be 2.

B = 2

# call the the class as a variable

ekf = EKFilter(dim_x=A, dim_y=B, dim_u=0)

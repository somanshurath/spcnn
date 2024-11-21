import numpy as np

# Cross
SE1 = np.zeros((3, 3), dtype=np.int32)
SE1[(0, 1, 2), (0, 1, 2)] = 1
SE1[(0, 1, 2), (2, 1, 0)] = 1

# Plus
SE2 = np.zeros((3, 3), dtype=np.int32)
SE2[1, :] = 1
SE2[:, 1] = 1

# Rhombus
SE3 = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.int32)
SE3[1, 1] = 0

# Square (Empty)
SE4 = np.ones((3, 3), dtype=np.int32)
SE4[1, 1] = 0

# Line to right
SE5 = np.zeros((3, 3), dtype=np.int32)
SE5[:, 2] = 1

# Line to left
SE6 = np.zeros((3, 3), dtype=np.int32)
SE6[:, 0] = 1

# Line to Top
SE7 = np.zeros((3, 3), dtype=np.int32)
SE7[0, :] = 1

# Line to Bottom
SE8 = np.zeros((3, 3), dtype=np.int32)
SE8[-1, :] = 1

SE = [SE1, SE2, SE3, SE4, SE5, SE6, SE7, SE8]
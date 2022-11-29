import numpy as np

def rotZ(t):
    """
    Rotation around Z-axis.
    """
    return np.array([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t), np.cos(t), 0],
        [0, 0, 1]
    ],dtype=object)


def rotY(t):
    """
    Rotation around Z-axis.
    """
    return np.array([
        [np.cos(t), 0, np.sin(t)],
        [0, 1, 0],
        [-np.sin(t),0,np.cos(t)]
    ],dtype=object)


def rotX(t):
    """
    Rotation around Z-axis.
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(t), -np.sin(t)],
        [0, np.sin(t), np.cos(t)]
    ],dtype=object)
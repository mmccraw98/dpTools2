import numpy as np

class Info:
    def __new__(cls, filepath, dtype=float):
        return np.loadtxt(filepath, dtype=dtype)
    
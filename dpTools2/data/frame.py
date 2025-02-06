import os
import numpy as np

class Frame:

    def __init__(self, path):
        if path is not None and os.path.exists(path):
            self.load_frame(path)
        else:
            self.variables = {}

    def load_frame(self, path):
        for filename in os.listdir(path):
            if filename.endswith('.dat'):
                var_name = filename.split('.')[0]
                data = np.loadtxt(os.path.join(path, filename))
                setattr(self, var_name, data)

    def __repr__(self):
        variables = {
            var: repr(getattr(self, var)) for var in dir(self)
            if not var.startswith('_') and not callable(getattr(self, var))
        }
        variables_repr = "\n\t".join(variables.keys())
        return f"Frame(\n\t{variables_repr}\n)"

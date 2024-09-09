import os
import numpy as np

class Configuration:

    def __init__(self, path):
        self.load_configuration(path)

    def load_configuration(self, path):
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
        return f"Configuration(\n\t{variables_repr}\n)"

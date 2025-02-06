import json

class Parameters:
    def __init__(self, filepath, name=None):
        self.name = name
        self.load_parameters(filepath)

    def load_parameters(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        for key, value in data.items():
            setattr(self, key, value)

    def __getattr__(self, key):
        if key in self.keys():
            return getattr(self, key)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        
    def __getitem__(self, key):
        return getattr(self, key)
    
    def keys(self):
        variables = [var for var in dir(self) if not var.startswith('_') and not callable(getattr(self, var))]
        return variables
    
    def values(self):
        return [getattr(self, var) for var in self.keys()]
    
    def items(self):
        return [(var, getattr(self, var)) for var in self.keys()]

    def __repr__(self):
        variables = {
            var: repr(getattr(self, var)) for var in dir(self)
            if not var.startswith('_') and not callable(getattr(self, var))
        }
        variables_repr = "\n\t".join(f"{k}={v}" for k, v in variables.items())
        return f"Parameters-{self.name}-(\n\t{variables_repr}\n)"

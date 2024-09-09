
class Parameters:
    def __init__(self, filepath):
        self.load_parameters(filepath)

    def load_parameters(self, filepath):
        with open(filepath, 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')
                try:
                    value = float(value)
                except:
                    pass
                setattr(self, key, value)

    def __repr__(self):
        variables = {
            var: repr(getattr(self, var)) for var in dir(self)
            if not var.startswith('_') and not callable(getattr(self, var))
        }
        variables_repr = "\n\t".join(f"{k}={v}" for k, v in variables.items())
        return f"Parameters(\n\t{variables_repr}\n)"

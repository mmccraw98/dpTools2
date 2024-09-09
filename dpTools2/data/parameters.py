
class Parameters:
    def __init__(self, filepath):
        self.data = self.load_parameters(filepath)

    def load_parameters(self, filepath):
        params = {}
        with open(filepath, 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')
                params[key] = value
        return params

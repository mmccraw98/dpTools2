import pandas as pd

class Metrics:
    def __new__(cls, filepath):
        return pd.read_csv(filepath, sep='\t')


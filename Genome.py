import numpy as np


class Genome:
    def __init__(self, _value):
        self.value = _value
        self.fitness = 0.0
        self.gain = None

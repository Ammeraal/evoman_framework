import numpy as np


class Genome:
    def __init__(self, _value, sigma = None):
        self.value = _value
        self.fitness = 0.0
        self.gain = None
        self.m_fitness = None
        self.m_p = None
        self.m_e = None

        self.sigma = sigma

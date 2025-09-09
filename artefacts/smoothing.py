import scipy
import numpy as np

from artefacts.base import BaseTransform


class SmoothingTransform(BaseTransform):

    def __init__(self, severity=3, params=None):
        super().__init__(severity, params)

    @property
    def default_params(self):
        return np.array([ 0.5, 4 ])

    @property
    def params_bounds(self):
        return [
            (0.1, 2), 
            (1,  10)
        ]

    def transform(self, img, severity=None):
        super().transform(img, severity)
        sigma = self.rng() * self.p[1] + self.p[0]
        return scipy.ndimage.gaussian_filter(img, sigma=sigma, truncate=10)
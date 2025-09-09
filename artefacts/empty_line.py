import random
import numpy as np

from artefacts.base import BaseTransform


class EmptyLineTransform(BaseTransform):

    def __init__(self, severity=3, params=None):
        super().__init__(severity, params)

    @property
    def default_params(self):
        return np.array([])

    @property
    def params_bounds(self):
        return []

    def transform(self, img, severity=None):
        super().transform(img, severity)
        h = img.shape[0]
        loc = int(random.random() * h / 2 + h / 4)
        img = img.copy()
        img[loc, :] = 0
        return img

import random
import numpy as np

from artefacts.base import BaseTransform


class LineShiftTransform(BaseTransform):

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
        par1_line_shift  = int(random.random() * h / 2 + h / 4)
        par4_entire_line = int(random.random() * 10 + 5)
        img = img.copy()
        img[par1_line_shift, par4_entire_line:] = img[par1_line_shift, :-par4_entire_line]
        return img

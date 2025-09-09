import random
import numpy as np
from numpy.fft import fft2, fftshift

from artefacts.base import BaseTransform
from artefacts import utils


class BadDataLineTransform(BaseTransform):

    def __init__(self, severity=3, params=None):
        super().__init__(severity, params)

    @property
    def default_params(self):
        return np.array([ 40 ])

    @property
    def params_bounds(self):
        return [ (10, 100) ]

    def transform(self, img, severity=None):
        super().transform(img, severity)
        
        h, w = img.shape
        par1_entire_line = (1 - self.rng()) * self.p[0]
        par2_entire_line = h / 4
        par3_entire_line = random.randrange(int(h / 2 - par2_entire_line), int(h / 2 + par2_entire_line))

        k_space = fftshift(fft2(img))
        k_space[par3_entire_line, :] += (utils.complex_rand_from_reference((w, ), k_space) / par1_entire_line)
        return utils.reconstruct_from_kspace(k_space)
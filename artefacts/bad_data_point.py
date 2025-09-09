import random
import numpy as np
from numpy.fft import fft2, fftshift

from artefacts.base import BaseTransform
from artefacts import utils


class BadDataPointTransform(BaseTransform):

    def __init__(self, severity=3, params=None):
        super().__init__(severity, params)

    @property
    def default_params(self):
        return np.array([ 1, 3, 4 ])

    @property
    def params_bounds(self):
        return [
            (0, 3), 
            (1, 5),
            (2, 5)
        ]

    def transform(self, img, severity=None):
        super().transform(img, severity)
        p = self.p

        h, w = img.shape
        par1_band = self.rng() * p[1] + p[0]
        par2_band = h / p[2]
        par3_band = w / p[2]
        par4_band = random.randrange(int(h / 2 - par2_band), int(h / 2 + par2_band))
        par5_band = random.randrange(int(w / 2 - par3_band), int(w / 2 + par3_band))

        k_space = fftshift(fft2(img))
        k_space[par4_band, par5_band] = utils.complex_rand_from_reference((1,), k_space) * par1_band
        return utils.reconstruct_from_kspace(k_space)

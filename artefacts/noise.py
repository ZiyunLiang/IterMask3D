import numpy as np
from numpy.fft import fft2, fftshift

from artefacts.base import BaseTransform
from artefacts import utils


class NoiseTransform(BaseTransform):

    def __init__(self, severity=3, params=None):
        super().__init__(severity, params)

    @property
    def default_params(self):
        return np.array([ 0.05 ])

    @property
    def params_bounds(self):
        return [ (0.01, 0.2) ]

    def transform(self, img, severity=None):
        super().transform(img, severity)

        # par1_noise = (self.rng() + self.p[0]) / 100
        # print('rng', self.rng(), par1_noise)
        par1_noise = 0.003

        h, w = img.shape
        k_space = fftshift(fft2(img))
        size_r = int(h - (h // 2) - 1)

        k_space[:size_r, :] = \
        k_space[:size_r, :] + utils.complex_uniform_ep((size_r, w), k_space, par1_noise)
        k_space[h - size_r:h, :] = \
        k_space[h - size_r:h, :] + utils.complex_uniform_ep((size_r, w), k_space, par1_noise)
        return utils.reconstruct_from_kspace(k_space)
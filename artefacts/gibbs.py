import numpy as np
from numpy.fft import fft2, fftshift

from artefacts.base import BaseTransform
from artefacts import utils


class GibbsTransform(BaseTransform):

    def __init__(self, severity=3, params=None):
        super().__init__(severity, params)

    @property
    def default_params(self):
        return np.array([ .04, .14 ])

    @property
    def params_bounds(self):
        return [
            (.01, .1),
            (.05, .20) 
        ]

    def transform(self, img, severity=None):
        super().transform(img, severity)
        p = self.p

        _min = p[0]
        _max = p[1]
        gam_h = (_max - _min) * (1 - self.rng() * 2) + _min
        gam_v = (_max - _min) * (1 - self.rng() * 2) + _min

        k_space = fftshift(fft2(img))
        k_space = self.gibbs_kspace_perturbation(k_space, gam_h, gam_v)
        recon_image = utils.reconstruct_from_kspace(k_space)
        return utils.sharpen(recon_image)
    

    def gibbs_kspace_perturbation(self,
                                  kspace: np.ndarray, 
                                  extension_perc_h: float, 
                                  extension_perc_v: bool):
        '''
        Perturbs image k-space by simulating the Gibbs artifact.
        '''
        k_space = kspace.copy()
        h, w = k_space.shape

        hT = int(h / 2 - h * extension_perc_h)
        hB = int(h / 2 + h * extension_perc_h)
        k_space[0:hT, 0:w] = complex(0.01, 0.01)
        k_space[hB:h, 0:w] = complex(0.01, 0.01)

        wT = int(w / 2 - w * extension_perc_v)
        wB = int(w / 2 + w * extension_perc_v)
        k_space[0:h, 0:wT] = complex(0.01, 0.01)
        k_space[0:h, wB:w] = complex(0.01, 0.01)

        return k_space



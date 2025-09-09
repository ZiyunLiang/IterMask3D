import numpy as np
from numpy.fft import fft2, fftshift
from skimage.exposure import rescale_intensity

from artefacts.base import BaseTransform
from artefacts import utils


class FoldingTransform(BaseTransform):

    def __init__(self, severity=3, params=None):
        super().__init__(severity, params)

    @property
    def default_params(self):
        return np.array([ 35, 2, 10 ])

    @property
    def params_bounds(self):
        return [  
            (10, 50), 
            (1,   5),
            (1,  10)
        ]

    def transform(self, img, severity=None):
        super().transform(img, severity)
        p = self.p
        spacing_size = int((1 - self.rng()) * p[0]) + int(p[1])
        min_number_of_central_bands = 3 + int(self.rng() * p[2]) 

        k_space = fftshift(fft2(img))
        h = img.shape[0]
        property_asel = np.zeros(h, dtype=bool)
        property_asel[0:h:spacing_size] = True
        property_asel[int(h / 2) - min_number_of_central_bands:int(h / 2) + min_number_of_central_bands] = False
        k_space = k_space[np.invert(property_asel), :]
        recon_image = utils.reconstruct_from_kspace(k_space)
        recon_image = rescale_intensity(recon_image, out_range=(img.min(), img.max()))
        recon_image = utils.pad_to_shape(recon_image, img.shape)
        return recon_image

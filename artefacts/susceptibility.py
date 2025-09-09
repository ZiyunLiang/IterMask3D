import scipy
import numpy as np
from skimage.exposure import rescale_intensity

from artefacts.base import BaseTransform
from artefacts import utils


class SusceptibilityTransform(BaseTransform):

    def __init__(self, severity=3, params=None):
        super().__init__(severity, params)

    @property
    def default_params(self):
        return np.array([ 0.3, 4])

    @property
    def params_bounds(self):
        return [ 
            (0, 1),
            (0, 7.5)
        ]

    def transform(self, img, severity=None):
        super().transform(img, severity)
        
        h, w = img.shape
        magnitude = (self.p[0] + (1-self.p[0]) * self.rng()) * min(h, w)
        smoothing = 2.5 + ( 7.5 - self.p[1] * self.rng() )

        out_img = rescale_intensity(img, out_range=np.uint8)
        out_img = self.rand_elastic_transform(out_img, magnitude, smoothing)        
        return rescale_intensity(utils.pad_to_shape(out_img, img.shape), out_range=np.uint8)


    def rand_elastic_transform(self, img, alpha, sigma):
        '''
        Elastic deformation of images as described in [Simard2003]_.
        [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        '''
        shape = img.shape
        randx = np.random.rand(*shape) * 2 - 1
        randy = np.random.rand(*shape) * 2 - 1
        dx = scipy.ndimage.gaussian_filter(randx, sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter(randy, sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        return scipy.ndimage.map_coordinates(img, indices, order=1).reshape(shape)


import os
import random
import numpy as np
import skimage

from artefacts.base import BaseTransform


class NoBrainTransform(BaseTransform):
    '''
    We don't need a transformation for the `no-brain` class, 
    but it is useful to implement the same BaseTransform class
    to stick to the same interface.
    '''

    def __init__(self, image_folder):
        super().__init__(0, None)
        assert os.path.exists(image_folder), 'invalid no-brain image folder path'
        self.image_folder = image_folder
        self.image_list = [ os.path.join(image_folder, f) for f in os.listdir(image_folder) ]

    @property
    def default_params(self):
        return np.array([])

    @property
    def params_bounds(self):
        return []

    def transform(self, img, severity=None):
        path = random.choice(self.image_list)
        return skimage.io.imread(path, as_gray=True)
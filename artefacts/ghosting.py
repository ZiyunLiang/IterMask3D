import random
import numpy as np
from numpy.fft import fft2, fftshift

from artefacts.base import BaseTransform
from artefacts.utils import rigid_warp, reconstruct_from_kspace


class GhostingTransform(BaseTransform):

    def __init__(self, severity=3, params=None):
        super().__init__(severity, params)

    @property
    def default_params(self):
        return np.array([ 10, 4, 10, 4, 26, 5, 3 ])

    @property
    def params_bounds(self):
        return [  
            (0, 99), 
            (0, 30),
            (0, 75),
            (0, 50),
            (0, 80),
            (1, 20),
            (1, 10)
        ]

    def transform(self, img, severity=None):
        super().transform(img, severity)
        p = self.p

        mrange                      = img.shape[1] * (self.rng() * p[0] + 1) / 100  
        min_lines_perc              = (p[1] + p[2] * self.rng()) / 100    
        tot_lines_perc              = (p[3] + p[4] * self.rng()) / 100    
        direction                   = random.choice([ 0, 1 ])
        mulfactor                   = p[5]
        permutations                = np.random.permutation(img.shape[direction])
        ang_rot                     = np.random.uniform(mrange) - mrange / 2
        tr_x                        = mrange * np.random.uniform() - mrange / 2
        tr_y                        = mrange * np.random.uniform() - mrange / 2
        min_number_of_central_bands = 3 + int(self.rng() * p[6]) 

        k_space = fftshift(fft2(img))
        curr_image_copy = img.copy()
        img_translation = rigid_warp(curr_image_copy, ang_rot, tr_x, tr_y)
        k_space_translated = fftshift(fft2(img_translation))
        
        dim_size = img.shape[direction]
        lines_to_edit = int((min_lines_perc + tot_lines_perc) * dim_size)
        lines_to_edit *= mulfactor

        k_space_motion = self._motion_mix(anchor_kspace=k_space, 
                                          motion_kspace=k_space_translated,
                                          number_of_lines=lines_to_edit,
                                          direction=direction, 
                                          direction_permutations=permutations, 
                                          central_bands_preserved=min_number_of_central_bands)

        recon_image = reconstruct_from_kspace(k_space_motion)
        return recon_image


    def _motion_mix(self, 
                    anchor_kspace,
                    motion_kspace,
                    number_of_lines,
                    direction,
                    direction_permutations, 
                    central_bands_preserved):
        '''
        anchor_kspace           : kspace of the original slice
        motion_kspace           : kspace of the moving slice
        number_of_lines         : number of lines in the kspace to replace
        direction               : 0 or 1 means respectively replacing horizontal or vertical lines
        direction_permutations  : permutation of the sequence [0,dim-1] where dim is choosen by direction 
        central_bands_preserved : amount of low-frequencies preserved
        
        returns a k-space containing the ghosting artifact.
        '''   
        number_of_lines = int(number_of_lines)
        anchor_kspace = anchor_kspace.copy()
        motion_kspace = motion_kspace.copy()
        dim_size = anchor_kspace.shape[direction]

        property_asel = np.zeros(dim_size, dtype=bool)
        property_asel[direction_permutations[:number_of_lines]] = True
        
        # ensure that central bands (low frequencies) are not modified
        fr = dim_size // 2 - central_bands_preserved
        to = dim_size // 2 + central_bands_preserved
        property_asel[fr:to] = False

        if direction == 0:
            anchor_kspace[np.invert(property_asel), :] = motion_kspace[np.invert(property_asel), :]
        else:
            anchor_kspace[:, np.invert(property_asel)] = motion_kspace[:, np.invert(property_asel)]

        return anchor_kspace
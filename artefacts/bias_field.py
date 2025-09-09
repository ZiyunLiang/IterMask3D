from typing import List, Sequence
import numpy as np

from artefacts.base import BaseTransform


class BiasFieldTransform(BaseTransform):

    def __init__(self, severity=3, params=None):
        super().__init__(severity, params)

    @property
    def default_params(self):
        return np.array([ 2, 5, .2, 0.5 ])

    @property
    def params_bounds(self):
        return [ 
            (1, 4), 
            (2, 9), 
            (0, 1), 
            (0, 1)
        ]

    def transform(self, img, severity=None):
        super().transform(img, severity)         
        bias_degree = int(self.p[0]) +  int(np.random.uniform() * self.p[1])
        lower_bound = int(self.p[2]) + self.rng() 
        upper_bound = lower_bound + self.p[3] + self.rng()

        bias_field = self.bias_field(img.shape, bias_degree, (lower_bound, upper_bound))
        return img * bias_field
        
        
    def bias_field(self, shape, degree, coeff_range=(1., .2), coeff=None):
        '''
        Generate a bias field.
        '''
        if coeff is None: coeff = self.random_poly_coeff(shape, degree, coeff_range)
        rank = len(shape)
        coeff_mat = np.zeros((degree + 1,) * rank)
        
        coords = [np.linspace(-1.0, 1.0, dim, dtype=np.float32) for dim in shape]
        prob = np.random.uniform()
        if prob >= .33 and prob < .66: coords[0] = np.linspace(1.0, -1.0, shape[0], dtype=np.float32)
        if prob >= .66 and prob <= 1.: coords[1] = np.linspace(1.0, -1.0, shape[1], dtype=np.float32)

        coeff_mat[np.tril_indices(degree + 1)] = coeff    
        field_exp = np.polynomial.legendre.leggrid2d(coords[0], coords[1], coeff_mat)
        return (1 / (1 + np.exp(field_exp)) )


    def random_poly_coeff(self, img_size: Sequence[int], degree, coeff_range) -> List:
        '''
        Generates random coeffiecients for Lagrange polynomials for bias on intensity. 
        '''
        n_coeff = int(np.prod([(degree + k) / k for k in range(1, len(img_size) + 1)]))
        return np.random.uniform(*coeff_range, n_coeff).tolist() # type: ignore

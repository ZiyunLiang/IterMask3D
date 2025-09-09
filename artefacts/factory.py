import numpy as np

from .folding import FoldingTransform
from .ghosting import GhostingTransform
from .gibbs import GibbsTransform
from .bad_data_point import BadDataPointTransform
from .smoothing import SmoothingTransform
from .zipper import ZipperTransform
from .noise import NoiseTransform
from .bad_data_line import BadDataLineTransform
from .susceptibility import SusceptibilityTransform
from .line_shift import LineShiftTransform
from .bias_field import BiasFieldTransform
from .empty_line import EmptyLineTransform


NAME_X_CLASS = {
    'folding':          FoldingTransform, 
    'motion':           GhostingTransform,
    'gibbs':            GibbsTransform, 
    'bdp':              BadDataPointTransform, 
    'smoothing':        SmoothingTransform, 
    'zipper':           ZipperTransform, 
    'noise':            NoiseTransform, 
    'bdl':              BadDataLineTransform,
    'susceptibility':   SusceptibilityTransform,
    'lineshift':        LineShiftTransform,
    'biasfield':        BiasFieldTransform,
    'emptyline':        EmptyLineTransform,
}

def load_class(name, severity=3, params_path=None):
    '''
    Load the artefact transformation class by name.
    '''
    global NAME_X_CLASS
    assert name in NAME_X_CLASS, 'invalid artefact name'
    transform = NAME_X_CLASS[name]()
    if params_path is not None and transform.get_n_params() > 0:
        params = np.load(params_path)
        transform.set_params(params)
    return transform
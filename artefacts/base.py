from abc import ABC, abstractmethod, abstractproperty
from artefacts.utils import powerlaw

class BaseTransform(ABC):

    def __init__(self, severity=1, params=None):
        self.p = params if params is not None else self.default_params
        self.severity = severity
        self.rng = lambda: powerlaw(0, 1, 10 - severity, 1)[0]

    def get_params(self):
        '''
        Return the transformation parameters
        '''
        return self.p
    
    def set_params(self, p):
        '''
        Set the transformation parameters
        '''
        self.p = p

    def get_n_params(self):
        '''
        Return the number of parameters
        '''
        return self.p.shape[0]

    def set_severity(self, severity):
        '''
        Set the severity of the transformation
        '''
        assert severity in range(0, 10), 'invalid severity parameter'
        self.severity = severity
        self.rng = lambda: powerlaw(0, 1, 10 - severity, 1)[0]

    @abstractproperty
    def default_params(self):
        '''
        Get default parameters
        '''
        pass

    @abstractproperty
    def params_bounds(self):
        '''
        Get parameters bounds for optimization
        '''
        pass


    @abstractmethod
    def transform(self, img, severity=None):
        '''
        Apply transformation to img
        '''
        if severity is not None: self.set_severity(severity)


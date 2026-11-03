import torch
from torchmfbd.regularization import Regularization

class RegularizationTime(Regularization):

    def __init__(self, lambda_reg=0.0, variable=None):
        super().__init__('time', variable, lambda_reg)
        
        self.lambda_reg = lambda_reg

    def __call__(self, x):
        # The time coupling is solved inside compute_object (filter_loss wiener_time*); this class is only a
        # placeholder and must not be added as a loss term that silently contributes zero
        raise NotImplementedError("RegularizationTime has no loss term: use filter_loss wiener_time with a `time` regularization entry")
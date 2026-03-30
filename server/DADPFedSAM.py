from client import dadpfedsam
from .DADPFed import DADPFed


class DADPFedSAM(DADPFed):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        super(DADPFedSAM, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        self.Client = dadpfedsam

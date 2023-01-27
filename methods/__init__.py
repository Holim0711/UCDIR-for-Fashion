from .iwcon import IWConModule
from .cwcon import CWConModule
from .selfent import SelfEntropyModule
from .distdist import DistDistModule
from .domadv import DomAdvModule


def get_module_class(name):
    if name == 'IWCon':
        return IWConModule
    if name == 'CWCon':
        return CWConModule
    if name == 'SelfEntropy':
        return SelfEntropyModule
    if name == 'DistDist':
        return DistDistModule
    if name == 'DomAdv':
        return DomAdvModule


def build_module(config):
    return get_module_class(config['method']['name'])(**config)


def load_module(config, checkpoint):
    return get_module_class(config['method']['name']).load_from_checkpoint(checkpoint)

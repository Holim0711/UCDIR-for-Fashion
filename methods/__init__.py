from .iwcon import IWConModule
from .cwcon import CWConModule
from .domadv import DomAdvModule


def build_module(config):
    if config['method']['name'] == 'IWCon':
        return IWConModule(**config)
    if config['method']['name'] == 'CWCon':
        return CWConModule(**config)
    if config['method']['name'] == 'DomAdv':
        return DomAdvModule(**config)


def load_module(config, checkpoint):
    if config['method']['name'] == 'IWCon':
        return IWConModule.load_from_checkpoint(checkpoint)
    if config['method']['name'] == 'CWCon':
        return CWConModule.load_from_checkpoint(checkpoint)
    if config['method']['name'] == 'DomAdv':
        return DomAdvModule.load_from_checkpoint(checkpoint)

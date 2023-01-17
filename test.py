import os
import sys
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.argparse import parse_env_variables
from torchvision import transforms as trfms

from methods import *
from datasets import *


def test(config, checkpoint):
    trainer = Trainer(**vars(parse_env_variables(Trainer)), logger=False)

    transform = trfms.Compose([
        trfms.Resize(256),
        trfms.CenterCrop(224),
        trfms.ToTensor(),
        trfms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transforms = {'source_train': None, 'target_train': None,
                  'source_val': transform, 'target_val': transform}
    batch_sizes = {'source_train': None, 'target_train': None,
                   'source_val': 32, 'target_val': 32}

    if config['dataset']['name'] == 'OfficeHome':
        dm = OfficeHomeDataModule(
            config['dataset']['root'],
            config['dataset']['query_domain'],
            config['dataset']['result_domain'],
            transforms=transforms,
            batch_sizes=batch_sizes)
    elif config['dataset']['name'] == 'DomainNet':
        dm = DomainNetDataModule(
            config['dataset']['root'],
            config['dataset']['query_domain'],
            config['dataset']['result_domain'],
            transforms=transforms,
            batch_sizes=batch_sizes)
    elif config['dataset']['name'] == 'DeepFashion':
        dm = DeepFashionDataModule(
            config['dataset']['root'],
            transforms=transforms,
            batch_sizes=batch_sizes)

    if config['method']['name'] == 'IWCon':
        model = IWConModule.load_from_checkpoint(checkpoint)

    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    logdir = sys.argv[1]
    hparams = os.path.join(logdir, 'hparams.yaml')
    print('hparams:', hparams)
    hparams = yaml.load(open(hparams), Loader=yaml.FullLoader)
    ckptdir = os.path.join(logdir, 'checkpoints')
    checkpoint = os.path.join(ckptdir, os.listdir(ckptdir)[0])
    print('checkpoint:', checkpoint)
    test(hparams, checkpoint)

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.argparse import parse_env_variables
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_lite import seed_everything
from torchvision import transforms as trfms

from methods import *
from datasets import *


@hydra.main(config_path='configs', version_base=None)
def main(config):
    seed_everything(config['random_seed'])

    trainer = Trainer(
        **vars(parse_env_variables(Trainer)),
        max_epochs=config['max_epochs'],
        callbacks=[
            ModelCheckpoint(monitor='val/mAP', mode='max'),
            LearningRateMonitor(),
        ]
    )

    train_transform = trfms.Compose([
        trfms.RandomResizedCrop(224, scale=(0.2, 1.)),
        trfms.RandomApply([trfms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        trfms.RandomGrayscale(p=0.2),
        trfms.RandomApply([trfms.GaussianBlur(5)], p=0.5),
        trfms.RandomHorizontalFlip(),
        trfms.ToTensor(),
        trfms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = trfms.Compose([
        trfms.Resize(256),
        trfms.CenterCrop(224),
        trfms.ToTensor(),
        trfms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transforms = {
        'source_train': trfms.Lambda(lambda x: (train_transform(x), train_transform(x))),
        'target_train': trfms.Lambda(lambda x: (train_transform(x), train_transform(x))),
        'source_val': val_transform,
        'target_val': val_transform,
    }
    batch_sizes = {
        'source_train': config['batch_size']['train'],
        'target_train': config['batch_size']['train'],
        'source_val': config['batch_size']['val'],
        'target_val': config['batch_size']['val'],
    }
    batch_sizes = {k: v // trainer.num_devices for k, v in batch_sizes.items()}

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
        model = IWConModule(**config)

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()

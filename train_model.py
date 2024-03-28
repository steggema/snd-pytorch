import os
import yaml
import click

import torch
try:
    from pytorch_lightning.loggers import WandbLogger
    import wandb
    use_wandb = True
except Exception:
    print("Wandb not installed")
    use_wandb = False

from pytorch_lightning import Trainer
from torch_geometric.data.lightning.datamodule import LightningDataset
from torch_geometric.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torchinfo import summary
import warnings
warnings.filterwarnings('ignore')

from LightningModules.Models.gravnet import GravNet
from LightningModules.Models.gravnetext import GravNetExt
from LightningModules.Models.exphormer import Exphormer

from Datasets.TopTagDataset import TopTagDataset
from Datasets.SNDDataset import SNDDataset

datasets = {
    'TopTag': TopTagDataset,
    'SND': SNDDataset
}

from pytorch_lightning.strategies import DDPStrategy


@click.command()
@click.argument('config', type=str, required=True)
@click.option('--root_dir', default=None)
@click.option('--checkpoint', default=None)

def main(config, root_dir=None, checkpoint=None):
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['root_dir'] = root_dir
    config['checkpoint'] = checkpoint
    train(config)

def train(config):

    if config["checkpoint"] is not None:
        loaded_configs = torch.load(config["checkpoint"])["hyper_parameters"]
        config.update(loaded_configs)

    model_name = config["model"]
    if model_name in globals():
        model = globals()[model_name](config)
    else:
        raise ValueError(f"Model name {model_name} not found in globals")

    os.makedirs(config["artifacts"], exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["artifacts"],
        filename='best',
        monitor="auc", 
        mode="max", 
        save_top_k=1, 
        save_last=True
    )
    
    if use_wandb:
        logger = WandbLogger(
            project=config["project"],
            save_dir=config["artifacts"],
        )
    else:
        logger = None

    if config["root_dir"] is None:
        if 'SLURM_JOB_ID' in os.environ:
            default_root_dir = os.path.join(".", os.environ['SLURM_JOB_ID'])
        else:
            default_root_dir = '.'
    else:
        default_root_dir = os.path.join(".", config["root_dir"])

    accelerator = "gpu" if torch.cuda.is_available() else None

    trainer = Trainer(
        accelerator = accelerator,
        devices="auto",
        # devices=1,
        num_nodes=config["nodes"],
        max_epochs=config["max_epochs"],
        logger=logger,
        # strategy=CustomDDPPlugin(find_unused_parameters=False),
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        callbacks=[checkpoint_callback],
        default_root_dir=default_root_dir
    )

    summary(model)
    
    
    dataset = DataLoader(datasets[config['dataset']]('/data2/steggema/snd/pt'), batch_size=config["train_batch"], shuffle=True, num_workers=1)
    # dataset = LightningDataset(datasets[config['dataset']]('/data2/steggema/toptagtest/'), batch_size=config["train_batch"])

    trainer.fit(model, dataset, ckpt_path=config["checkpoint"])


if __name__ == "__main__":
    main()

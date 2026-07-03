import torch
import lightning as pl
import utils.config as load_config
import argparse
import yaml

from torch.utils.data import DataLoader
from DisGlioma.src.utils.dataset_vision import Universal
from DisGlioma.src.wrapper import DisGliomaWrapper
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping
from pathlib import Path


def get_parser(yaml_config:str=None):
    with open(yaml_config, 'r') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description='Training DisGlioma')
    
    parser.add_argument('--config', default=yaml_config, type=str)
    
    parser.add_argument('--train_bsz', default=config['TRAIN']['train_batch_size'], type=int)
    parser.add_argument('--valid_bsz', default=config['TRAIN']['valid_batch_size'], type=int)
    parser.add_argument('--root_dir', default=config['DATA']['root_dir'], type=str)
    parser.add_argument('--csv_path', default=config['DATA']['csv_path'], type=str)
    parser.add_argument('--fold', default=config['DATA']['fold'], type=int)
    parser.add_argument('--save_path', default=config['MODEL']['model_save_path'], type=str)

    args = parser.parse_args()
    assert args.config is not None

    cfg = load_config.load_cfg_from_cfg_file(args.config)

    return args, cfg


if __name__ == '__main__':

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    cfg_path = (PROJECT_ROOT / "src" / "config" / "DisGlioma.yaml").resolve()

    args, cfg = get_parser()
    print("cuda:",torch.cuda.is_available())

    save_path = (PROJECT_ROOT / args.save_path).resolve()
    root_dir = (PROJECT_ROOT / args.root_dir).resolve()
    csv_path = (PROJECT_ROOT / args.csv_path).resolve()

    ds_train = Universal(root_dir, csv_path, mode='train', ds_name=None, fold=args.fold).cache_dataset()
    ds_valid = Universal(root_dir, csv_path, mode='valid', ds_name=None, fold=args.fold).cache_dataset()
    
    dl_train = DataLoader(ds_train, batch_size=args.train_bsz, shuffle=True, num_workers=args.train_bsz)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_bsz, shuffle=False, num_workers=args.valid_bsz)

    model = DisGliomaWrapper(cfg)

    ## 1. setting recall function
    model_ckpt = ModelCheckpoint(
        dirpath=save_path,
        filename="disglioma",
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        verbose=True,
        save_weights_only=True
    )
    model_ckpt.FILE_EXTENSION = ".pt"

    early_stopping = EarlyStopping(monitor = 'val_loss',
                            patience=cfg.patience,
                            mode = 'min'
    )

    ## 2. setting trainer
    trainer = pl.Trainer(logger=None,
                        min_epochs=cfg.min_epochs,max_epochs=cfg.max_epochs,
                        accelerator='gpu', 
                        devices=1,
                        callbacks=[model_ckpt,early_stopping],
                        ) 

    ## 3. start training
    print('start training')
    trainer.fit(model,dl_train,dl_valid,)
    print('done training')


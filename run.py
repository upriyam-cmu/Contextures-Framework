# -*- coding: utf-8 -*-
'''
Usage:
$ python run.py
$ python run.py config=my.yaml
'''

from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import yaml
from yaml import safe_load
from easydict import EasyDict as edict
from sklearn.preprocessing import LabelEncoder

from scripts.loader import load_dataset
from feature_transforms.pipeline import ColumnPipeline
from utils.registry import get_encoder, get_loss, get_context
from trainer.trainer import SVDTrainer
from downstream import run_probe

# Helper functions
def load_cfg(path: str | Path) -> edict:
    with open(path, 'r') as f:
        return edict(yaml.safe_load(f))


def build_feature_pipeline(fp_cfg: Dict[str, Any]) -> ColumnPipeline:
    if fp_cfg.get('name', '').lower() == 'identity':
        return ColumnPipeline()
    return ColumnPipeline(numeric = fp_cfg.get('numeric', []),
                          categorical = fp_cfg.get('categorical', []))


def train_val_test_split(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
    n = len(X)
    perm = np.random.permutation(n)
    n_train = int(0.8 * n)
    n_val   = int(0.9 * n)
    tr, va, te = perm[:n_train], perm[n_train:n_val], perm[n_val:]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


# Main
def main(cfg: edict) -> None:
    torch.manual_seed(cfg["global"]["seed"])
    np.random.seed(cfg["global"]["seed"])
    device = torch.device(cfg["global"]["device"])

    results_dir = Path(cfg["global"]["results_dir"])
    (results_dir / "models").mkdir(parents = True, exist_ok = True)

    # single tag / group
    tags: list[str]
    if 'tag' in cfg.dataset:
        tags = [cfg.dataset.tag]
    else:
        big_yaml = safe_load(open('configs/datasets.yaml'))
        tags = big_yaml[cfg.dataset.group]

    for tag in tags:
        print(f'\n--- {tag} ---')

        # load raw
        X_df, y, meta = load_dataset(tag)
        if meta['target_type'] in ('classification', 'binary'):
            le = LabelEncoder()
            y = le.fit_transform(y)

        meta['path'] = str(Path('data') / tag)  # let split helper locate split file

        Xtr_raw, ytr, Xva_raw, yva, Xte_raw, yte = train_val_test_split(X_df.values, y)

        Xtr_df = pd.DataFrame(Xtr_raw, columns = X_df.columns)
        Xva_df = pd.DataFrame(Xva_raw, columns = X_df.columns)
        Xte_df = pd.DataFrame(Xte_raw, columns = X_df.columns)

        # feature pipeline
        pipe = build_feature_pipeline(cfg.feature_preprocessing)
        Xtr = pipe.fit_transform(Xtr_df)
        Xva = pipe.transform(Xva_df)
        Xte = pipe.transform(Xte_df)

        # context
        CtxCls = get_context(cfg.context.name)
        context = CtxCls(**cfg.context.parameters)
        context.fit(Xtr_df)
        context_collate = context.get_collate_fn()

        # datasets / loaders
        train_bs = int(cfg['train']['batch_size'])
        def make_loader(X: pd.DataFrame, *, shuffle = False):
            x_t = torch.tensor(X.values, dtype = torch.float32)
            ds = TensorDataset(x_t)

            def my_collate(batch):
                # batch -> list[tuple[tensor]]
                x_batch = torch.stack([b[0] for b in batch], dim = 0)
                return context_collate(x_batch)
                    
            return DataLoader(ds, batch_size = train_bs, shuffle = shuffle, collate_fn = my_collate)
        
        train_loader = make_loader(Xtr, shuffle = True)

        # encoders
        EncCls   = get_encoder(cfg.encoder.name)
        x_enc = EncCls(input_dim = pipe.output_dim, **cfg.encoder.parameters).to(device)
        a_enc = EncCls(input_dim = context.context_dim, **cfg.encoder.parameters).to(device)

        # optimizer & loss
        params = list(x_enc.parameters()) + list(a_enc.parameters())
        optim = torch.optim.Adam(params, lr = float(cfg['train']['lr']))
        LossCls = get_loss(cfg.losses.name)
        criterion = LossCls(**cfg.losses.parameters)

        # trainer
        trainer = SVDTrainer(x_enc, a_enc, optim, criterion,
                             train_loader, cfg.train.num_epochs,
                             device = device)
        trainer.train()
        trainer.save_model(results_dir / 'models')

        # frozen probe
        for p in x_enc.parameters():
            p.requires_grad_(False)

        print('Extracting train/val/test features...')
        def feats(X):
            t = torch.tensor(X.values, dtype=torch.float32, device = device)
            with torch.no_grad():
                return x_enc(t).cpu()
            
        f_tr, f_va, f_te = feats(Xtr), feats(Xva), feats(Xte)

        probe_kind = cfg['probe']['kind']
        probe_params = cfg['probe'].get('params', {})

        probe, probe_res = run_probe(
            kind        = probe_kind,
            features    = f_tr,
            targets     = ytr,
            task_type   = meta['target_type'],
            X_val       = f_va, y_val = yva,
            X_test      = f_te, y_test = yte,
            **probe_params # defined in config
        )

        # summary
        print('Probe test metrics:', probe_res['test_metrics'])

        out_json = results_dir / f'{tag}__full.json'
        with open(out_json,'w') as f:
            json.dump({
                'meta': meta,
                'probe': probe_res,
                'config': cfg,
            }, f, indent=2)
        print('Saved:', out_json)


# -------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='configs/config.yaml',
                        help='Path to YAML experiment file')
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    main(cfg)
import tsl
import torch
import numpy as np
from tsl.datasets import MetrLA, PemsBay
import argparse
from tsl.data import SpatioTemporalDataset

from tsl.data import SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler

from torch import nn

from tsl.nn.models.transformer_model import TransformerModel
from tsl.nn.models.graphormer_model import GraphormerModel, discretize_values, Graph

from tsl.nn.metrics.metrics import MaskedMAE, MaskedMAPE
from tsl.predictors import Predictor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from tsl.ops.connectivity import edge_index_to_adj
import matplotlib.pyplot as plt
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-name", type=str, default='la')
parser.add_argument("--model", type= str, default = 'graphormer')
parser.add_argument("--horizon", type=int, default=12)
parser.add_argument("--window", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--val_len", type=float, default=0.1)
parser.add_argument("--test_len", type=float, default=0.2)
parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--ff_size", type=int, default=64)
parser.add_argument("--n_heads", type=int, default=2)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--axis", type=str, default='both')
parser.add_argument("--activation", type=str, default='elu')
parser.add_argument("--sge", type=bool, default=False)
parser.add_argument("--ce", type=bool, default=False)
args = parser.parse_args()

if args.dataset_name == 'la':
    dataset = MetrLA()
elif args.dataset_name == 'bay':
    dataset = PemsBay()
else:
    raise ValueError(f"Dataset {args.dataset_name} not available in this setting.")

if (1 - (args.val_len + args.test_len)) < 0:
    raise ValueError(f"Validation and Testing dataset split ratios should be less than 1")

if args.model == 'transformer':
    model = TransformerModel
elif args.model == 'graphormer':
    model = GraphormerModel
    assert args.axis == 'both', "Graphormer currently supports axis='both' only"

df, dist, mask = dataset.load()
dist[dist == float('inf')] = 0
dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))

adj = dataset.get_connectivity(threshold=0.1,
                               include_self=False,
                               normalize_axis=1,
                               layout="edge_index")

edge_index, edge_weight = adj

dense = edge_index_to_adj(edge_index, edge_weight)
connectivity_matrix = torch.from_numpy(dense)
binarized_matrix = torch.where(connectivity_matrix > 0.0, 1., 0.)

in_degree = torch.sum(binarized_matrix, axis=0).int()
out_degree = torch.sum(binarized_matrix, axis=1).int()
discretized_matrix = discretize_values(dist)

g = Graph(discretized_matrix.cpu().detach().numpy())
spd = np.zeros_like(discretized_matrix)
for u in range(spd.shape[0]):
   dist = g.dijkstra(u, 0)
   spd[u, :] = dist

spd_matrix = torch.from_numpy(spd).int()

torch_dataset = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                                      connectivity=adj,
                                      mask=dataset.mask,
                                      horizon=args.horizon,
                                      window=args.window)

BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.n_epochs
scalers = {'data': StandardScaler(axis=(0, 1))}
splitter = dataset.get_splitter(val_len=args.val_len, test_len=args.test_len)

dm = SpatioTemporalDataModule(
    dataset=torch_dataset,
    scalers=scalers,
    splitter=splitter,
    batch_size=BATCH_SIZE,
    workers=4
)

dm.setup()
loss_fn = MaskedMAE(compute_on_step=True)


metrics = {'mae': MaskedMAE(compute_on_step=False),
           'mape': MaskedMAPE(compute_on_step=False),
           'mae_at_15': MaskedMAE(compute_on_step=False, at=2),  # `2` indicated the third time step,
                                                                 # which correspond to 15 minutes ahead
           'mae_at_30': MaskedMAE(compute_on_step=False, at=5),
           'mae_at_60': MaskedMAE(compute_on_step=False, at=11), }

if model == TransformerModel:
    model_kwargs = {
        'input_size': dm.n_channels,  # 1 channel
        'hidden_size': 64,
        'output_size': dm.n_channels,
        'ff_size': 64,
        'exog_size': 0,
        'horizon': dm.horizon,  # 12, the number of steps ahead to forecast
        'n_heads': 2,
        'n_layers': 2,
        'n_nodes' : dm.n_nodes,
        'max_in_degree': max(in_degree) + 1,
        'max_out_degree': max(out_degree) + 1,
        'in_degree_list': in_degree,
        'out_degree_list': out_degree,
        'dropout': 0,
        'axis': args.axis,  #Must be 'both'
        'activation': args.activation,
        'with_sge' : args.sge,
        'with_ce' : args.ce
    }
elif model == GraphormerModel:
    model_kwargs = {
    'input_size': dm.n_channels,  # 1 channel
    'hidden_size': 64,
    'output_size': dm.n_channels,
    'ff_size': 64,
    'exog_size': 0,
    'horizon': dm.horizon,  # 12, the number of steps ahead to forecast
    'n_nodes': dm.n_nodes,
    'n_heads': 2,
    'n_layers': 2,
    'max_in_degree': max(in_degree) + 1,
    'max_out_degree': max(out_degree) + 1,
    'max_dist': torch.max(spd_matrix) + 1,
    'in_degree_list': in_degree,
    'out_degree_list': out_degree,
    'spd_matrix': spd_matrix,
    'dropout': 0.,
    'axis': args.axis, #Must be 'steps'
    'activation': args.activation
}


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
scheduler_kwargs = {
    'T_max': int(24648 / BATCH_SIZE) * NUM_EPOCHS
}

# setup predictor
predictor = Predictor(
    model_class=model,
    model_kwargs=model_kwargs,
    optim_class=torch.optim.AdamW,
    optim_kwargs={'lr': 5e-4},
    loss_fn=loss_fn,
    metrics=metrics,
    )

checkpoint_callback = ModelCheckpoint(
    dirpath='logs',
    save_top_k=1,
    monitor='val_mae',
    mode='min',
)

trainer = pl.Trainer(max_epochs=args.n_epochs,
                     gpus=1 if torch.cuda.is_available() else None,
                     limit_train_batches=1.0,
                     callbacks=[checkpoint_callback],
                     accumulate_grad_batches=1)

trainer.fit(predictor, datamodule=dm)

predictor.load_model(checkpoint_callback.best_model_path)
predictor.freeze()

performance = trainer.test(predictor, datamodule=dm)


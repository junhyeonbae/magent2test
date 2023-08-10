import dgl
import numpy as np
import torch as th
from dgl.nn import DenseSAGEConv

feat = th.ones(6, 10)
print(feat)
adj = th.tensor([[0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0.]])
conv = DenseSAGEConv(10, 2)
res = conv(adj, feat)
print(res)